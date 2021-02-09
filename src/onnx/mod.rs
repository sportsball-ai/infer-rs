use scopeguard::ScopeGuard;
use std::{
    ffi::{c_void, CStr, CString},
    fmt,
    marker::PhantomData,
    os::{raw::c_uint, unix::ffi::OsStrExt},
    path::Path,
};

macro_rules! c_str {
    ($s:expr) => {{
        concat!($s, "\0").as_ptr() as *const i8
    }};
}

mod api;
use api::*;

mod sys;

#[derive(Debug)]
pub struct Error {
    pub code: c_uint,
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

impl std::error::Error for Error {}

struct MemoryInfo {
    api: API,
    inner: *mut sys::OrtMemoryInfo,
}

impl Drop for MemoryInfo {
    fn drop(&mut self) {
        unsafe { self.api.release_memory_info(self.inner) }
    }
}

pub struct Environment {
    api: API,
    memory_info: MemoryInfo,
    inner: *mut sys::OrtEnv,
}

#[derive(thiserror::Error, Debug)]
pub enum NewEnvironmentError {
    #[error("unsupported api version")]
    UnsupportedAPIVersion,
    #[error(transparent)]
    Other(#[from] Error),
}

impl From<NewAPIError> for NewEnvironmentError {
    fn from(err: NewAPIError) -> Self {
        match err {
            NewAPIError::UnsupportedVersion => Self::UnsupportedAPIVersion,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum NewSessionError {
    #[error("malformed model path")]
    MalformedModelPath,
    #[error(transparent)]
    Other(#[from] Error),
}

impl Environment {
    pub fn new() -> Result<Environment, NewEnvironmentError> {
        let api = API::new()?;
        unsafe {
            let memory_info = MemoryInfo {
                api,
                inner: api.create_cpu_memory_info(
                    sys::OrtAllocatorType_OrtArenaAllocator,
                    sys::OrtMemType_OrtMemTypeDefault,
                )?,
            };
            let inner =
                api.create_env(sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING, c_str!(""))?;
            Ok(Environment {
                api,
                inner,
                memory_info,
            })
        }
    }

    pub fn new_session<P: AsRef<Path>>(
        &self,
        model_path: P,
    ) -> Result<Session<'_>, NewSessionError> {
        let model_path = CString::new(model_path.as_ref().as_os_str().as_bytes())
            .map_err(|_| NewSessionError::MalformedModelPath)?;
        unsafe {
            let allocator = self.api.get_allocator_with_default_options()?;
            let sess_options = self.api.create_session_options()?;

            sys::OrtSessionOptionsAppendExecutionProvider_CUDA(sess_options, 0);
            let sess = scopeguard::guard(
                self.api
                    .create_session(self.inner, model_path.as_ptr(), sess_options)?,
                |ptr| self.api.release_session(ptr),
            );

            let output_c_names = (0..self.api.session_get_output_count(*sess)?)
                .map(|i| -> Result<CString, Error> {
                    let raw = self.api.session_get_output_name(*sess, i, allocator)?;
                    let ret = CStr::from_ptr(raw).to_owned();
                    self.api.allocator_free(allocator, raw as _)?;
                    Ok(ret)
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let output_c_name_ptrs = output_c_names.iter().map(|s| s.as_ptr()).collect();

            let output_names = output_c_names
                .iter()
                .map(|s| {
                    s.as_c_str()
                        .to_str()
                        .expect("ort shouldn't return invalid names")
                        .to_string()
                })
                .collect();

            Ok(Session {
                api: self.api,
                inner: ScopeGuard::into_inner(sess),
                output_names,
                _output_c_names: output_c_names,
                output_c_name_ptrs,
                env: self,
            })
        }
    }

    pub fn new_tensor<'t, 'a: 't, 'data: 't, T: DataType>(
        &'a self,
        data: &'data [T],
        shape: &[usize],
    ) -> Result<Tensor<'t>, Error> {
        let ort_shape: Vec<_> = shape.iter().map(|n| *n as i64).collect();
        let data_type = T::tensor_element_data_type();
        unsafe {
            Ok(Tensor {
                api: self.api,
                inner: self.api.create_tensor_with_data_as_ort_value(
                    self.memory_info.inner,
                    data.as_ptr() as _,
                    (data.len() * std::mem::size_of::<T>()) as _,
                    &ort_shape,
                    data_type,
                )?,
                data_type,
                data_ptr: data.as_ptr() as _,
                shape: shape.to_vec(),
                env_and_data: PhantomData,
            })
        }
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        unsafe { self.api.release_env(self.inner) }
    }
}

pub struct Session<'env> {
    api: API,
    inner: *mut sys::OrtSession,
    output_names: Vec<String>,
    _output_c_names: Vec<CString>,
    output_c_name_ptrs: Vec<*const ::std::os::raw::c_char>,
    env: &'env Environment,
}

#[derive(thiserror::Error, Debug)]
pub enum SessionRunError {
    #[error("malformed input name")]
    MalformedInputName,
    #[error(transparent)]
    Other(#[from] Error),
}

impl<'env> Session<'env> {
    pub fn environment(&self) -> &Environment {
        self.env
    }

    pub fn run(
        &self,
        inputs: &[(&str, Tensor)],
    ) -> Result<Vec<(&str, Tensor<'env>)>, SessionRunError> {
        let input_names: Vec<CString> = inputs
            .iter()
            .map(|(name, _)| {
                CString::new(name.as_bytes()).map_err(|_| SessionRunError::MalformedInputName)
            })
            .collect::<Result<_, SessionRunError>>()?;
        let input_name_ptrs: Vec<*const ::std::os::raw::c_char> = input_names
            .iter()
            .map(|name| name.as_c_str().as_ptr())
            .collect();
        let input_ptrs: Vec<_> = inputs
            .iter()
            .map(|(_, input)| input.inner as *const sys::OrtValue)
            .collect();
        unsafe {
            let outputs = self.api.run(
                self.inner,
                std::ptr::null(),
                &input_name_ptrs,
                &input_ptrs,
                &self.output_c_name_ptrs,
            )?;
            let outputs: Vec<_> = outputs
                .into_iter()
                .map(|ptr| scopeguard::guard(ptr, |ptr| self.api.release_value(ptr)))
                .collect();
            Ok(outputs
                .into_iter()
                .enumerate()
                .map(|(i, value)| -> Result<(&str, Tensor), Error> {
                    let info =
                        scopeguard::guard(self.api.get_tensor_type_and_shape(*value)?, |ptr| {
                            self.api.release_tensor_type_and_shape_info(ptr)
                        });
                    let data_type = self.api.get_tensor_element_type(*info)?;
                    let data_ptr = self.api.get_tensor_mutable_data(*value)?;
                    let dims = self.api.get_dimensions_count(*info)?;
                    let mut dims = vec![0; dims as usize];
                    self.api.get_dimensions(*info, &mut dims)?;
                    let shape = dims.into_iter().map(|n| n as _).collect();
                    Ok((
                        &self.output_names[i],
                        Tensor {
                            api: self.api,
                            inner: ScopeGuard::into_inner(value),
                            data_type,
                            data_ptr: data_ptr,
                            shape: shape,
                            env_and_data: PhantomData,
                        },
                    ))
                })
                .collect::<Result<_, Error>>()?)
        }
    }
}

impl<'env> Drop for Session<'env> {
    fn drop(&mut self) {
        unsafe { self.api.release_session(self.inner) }
    }
}

pub trait DataType {
    fn tensor_element_data_type() -> sys::ONNXTensorElementDataType;
}

impl DataType for f32 {
    fn tensor_element_data_type() -> sys::ONNXTensorElementDataType {
        sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    }
}

pub struct Tensor<'a> {
    api: API,
    inner: *mut sys::OrtValue,
    data_type: sys::ONNXTensorElementDataType,
    data_ptr: *const c_void,
    shape: Vec<usize>,
    env_and_data: PhantomData<&'a ()>,
}

impl<'a> Tensor<'a> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn as_slice<T: DataType>(&self) -> Option<&'a [T]> {
        if T::tensor_element_data_type() == self.data_type {
            unsafe {
                Some(std::slice::from_raw_parts(
                    self.data_ptr as _,
                    self.shape.iter().product(),
                ))
            }
        } else {
            None
        }
    }
}

impl<'a> Drop for Tensor<'a> {
    fn drop(&mut self) {
        unsafe { self.api.release_value(self.inner) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // This test verifies that basic functionality works by running upsample.onnx, which was
    // produced via:
    //
    // ```
    // import subprocess
    // from tensorflow import keras
    //
    // m = keras.Sequential([
    //     keras.layers.UpSampling2D(size=2)
    // ])
    // m.build(input_shape=(None, None, None, 3))
    // m.summary()
    // m.save('saved_model')
    //
    // subprocess.check_call([
    //     'python', '-m', 'tf2onnx.convert',
    //     '--saved-model', 'saved_model',
    //     '--opset', '12',
    //     '--output', 'upsample.onnx',
    // ])
    // ```
    #[test]
    fn test_upsample() {
        let env = Environment::new().unwrap();
        let sess = env.new_session("src/onnx/testdata/upsample.onnx").unwrap();
        let input = array![[1., 2., 3.], [3., 4., 5.]];
        let input = env
            .new_tensor(input.as_slice().unwrap(), &[1, 1, 2, 3])
            .unwrap();
        let outputs = sess.run(&[("up_sampling2d_input:0", input)]).unwrap();
        assert_eq!(outputs.len(), 1);
        let (name, output) = &outputs[0];
        assert_eq!(name, &"Identity:0");
        assert_eq!(output.shape(), vec![1, 2, 4, 3]);
        assert_eq!(
            output.as_slice::<f32>().unwrap(),
            vec![
                1., 2., 3., 1., 2., 3., 3., 4., 5., 3., 4., 5., 1., 2., 3., 1., 2., 3., 3., 4., 5.,
                3., 4., 5.
            ]
        );
    }
}
