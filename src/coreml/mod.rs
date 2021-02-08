use std::{
    ffi::{c_void, CStr, CString},
    os::unix::ffi::OsStrExt,
    path::Path,
};

mod sys {
    use std::ffi::c_void;

    extern "C" {
        pub fn open_coreml_model(path: *const i8) -> *const c_void;
        pub fn mlmodel_predict(
            model: *const c_void,
            input_names: *const *const i8,
            input_data_ptrs: *const *const f32,
            input_dimensionalities: *const u32,
            input_shapes: *const *const u32,
            input_count: u32,
        ) -> *const c_void;
        pub fn mlmultiarray_dimensionality(multiarray: *const c_void) -> u32;
        pub fn mlmultiarray_get_shape(multiarray: *const c_void, dest: *mut u32);
        pub fn mlmultiarray_data(multiarray: *const c_void) -> *const c_void;
        pub fn mlmodel_output_count(model: *const c_void) -> u32;
        pub fn mlmodel_get_output_names(model: *const c_void, dest: *mut *const c_void);
        pub fn mlfeatureprovider_multiarray_by_name(
            provider: *const c_void,
            name: *const i8,
        ) -> *const c_void;
        pub fn nsstring_utf8(s: *const c_void) -> *const i8;
        pub fn release_object(obj: *const c_void);
    }
}

pub struct MLModel {
    inner: *const c_void,
    output_names: Vec<String>,
    output_c_names: Vec<CString>,
}

#[derive(thiserror::Error, Debug)]
pub enum NewMLModelError {
    #[error("malformed path")]
    MalformedPath,
    #[error("open error")]
    OpenError,
}

#[derive(thiserror::Error, Debug)]
pub enum PredictError {
    #[error("malformed input name")]
    MalformedInputName,
    #[error("predict error")]
    PredictError,
}

pub struct InputTensor<'a> {
    pub data: &'a [f32],
    pub shape: &'a [usize],
}

struct OutputProvider(*const c_void);

impl OutputProvider {
    fn output_tensor(&self, name: &CStr) -> Option<OutputTensor> {
        unsafe {
            let multiarray = sys::mlfeatureprovider_multiarray_by_name(self.0, name.as_ptr());
            if multiarray.is_null() {
                None
            } else {
                let dimensionality = sys::mlmultiarray_dimensionality(multiarray);
                let mut shape = vec![0; dimensionality as usize];
                sys::mlmultiarray_get_shape(multiarray, shape.as_mut_ptr());
                Some(OutputTensor {
                    multiarray,
                    shape: shape.into_iter().map(|n| n as usize).collect(),
                })
            }
        }
    }
}

impl Drop for OutputProvider {
    fn drop(&mut self) {
        unsafe { sys::release_object(self.0) }
    }
}

pub struct OutputTensor {
    multiarray: *const c_void,
    shape: Vec<usize>,
}

impl OutputTensor {
    pub fn as_slice(&self) -> &[f32] {
        unsafe {
            let ptr = sys::mlmultiarray_data(self.multiarray);
            std::slice::from_raw_parts(ptr as _, self.shape.iter().product())
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl Drop for OutputTensor {
    fn drop(&mut self) {
        unsafe { sys::release_object(self.multiarray) }
    }
}

impl MLModel {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, NewMLModelError> {
        let path = match CString::new(path.as_ref().as_os_str().as_bytes()) {
            Ok(s) => s,
            Err(_) => return Err(NewMLModelError::MalformedPath),
        };
        unsafe {
            let ptr = sys::open_coreml_model(path.as_ptr());
            if ptr.is_null() {
                Err(NewMLModelError::OpenError)
            } else {
                let output_count = sys::mlmodel_output_count(ptr);
                let mut output_names = vec![std::ptr::null(); output_count as usize];
                sys::mlmodel_get_output_names(ptr, output_names.as_mut_ptr());
                let (output_names, output_c_names) = output_names
                    .into_iter()
                    .map(|s| {
                        let c_str = CStr::from_ptr(sys::nsstring_utf8(s)).to_owned();
                        sys::release_object(s);
                        let name = c_str
                            .to_str()
                            .expect("coreml should never return invalid strings")
                            .to_string();
                        (name, c_str)
                    })
                    .unzip();
                Ok(Self {
                    inner: ptr,
                    output_names,
                    output_c_names,
                })
            }
        }
    }

    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    pub fn predict(
        &self,
        inputs: &[(&str, InputTensor)],
    ) -> Result<Vec<(&str, OutputTensor)>, PredictError> {
        let input_names: Vec<_> = match inputs
            .iter()
            .map(|(name, _)| CString::new(name.as_bytes()))
            .collect::<Result<_, _>>()
        {
            Ok(names) => names,
            Err(_) => return Err(PredictError::MalformedInputName),
        };
        let input_name_ptrs: Vec<_> = input_names.iter().map(|name| name.as_ptr()).collect();
        let input_data_ptrs: Vec<_> = inputs
            .iter()
            .map(|(_, input)| input.data.as_ptr())
            .collect();
        let input_dimensionalities: Vec<_> = inputs
            .iter()
            .map(|(_, input)| input.shape.len() as u32)
            .collect();
        let input_shapes: Vec<Vec<u32>> = inputs
            .iter()
            .map(|(_, input)| input.shape.iter().map(|n| *n as u32).collect())
            .collect();
        let input_shapes: Vec<_> = input_shapes.iter().map(|shape| shape.as_ptr()).collect();
        let output_provider = unsafe {
            sys::mlmodel_predict(
                self.inner,
                input_name_ptrs.as_ptr(),
                input_data_ptrs.as_ptr(),
                input_dimensionalities.as_ptr(),
                input_shapes.as_ptr(),
                inputs.len() as _,
            )
        };
        if output_provider.is_null() {
            return Err(PredictError::PredictError);
        }
        let output_provider = OutputProvider(output_provider);
        let mut outputs = vec![];
        for (name, c_name) in self.output_names.iter().zip(&self.output_c_names) {
            if let Some(output) = output_provider.output_tensor(&c_name) {
                outputs.push((name.as_str(), output));
            }
        }
        Ok(outputs)
    }
}

impl Drop for MLModel {
    fn drop(&mut self) {
        unsafe { sys::release_object(self.inner) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // This test verifies that basic functionality works by running upsample.mlmodel, which was
    // produced via:
    //
    // ```
    // import coremltools as ct
    // from tensorflow import keras
    //
    // m = keras.Sequential([
    //     keras.layers.UpSampling2D(size=2)
    // ])
    // m.build(input_shape=(None, 1, 2, 3))
    // m.summary()
    // m.save('saved_model')
    //
    // mlmodel = ct.convert(m)
    // mlmodel.save('upsample.mlmodel')
    // ```
    #[test]
    fn test_upsample() {
        let model = MLModel::new("src/coreml/testdata/upsample.mlmodel").unwrap();
        assert_eq!(model.output_names(), &["Identity"]);

        let input = array![[1., 2., 3.], [3., 4., 5.]];
        let input = InputTensor {
            data: input.as_slice().unwrap(),
            shape: &[1, 1, 2, 3],
        };
        let outputs = model.predict(&[("up_sampling2d_input", input)]).unwrap();
        assert_eq!(outputs.len(), 1);
        let (name, output) = &outputs[0];
        assert_eq!(name, &"Identity");
        assert_eq!(output.shape(), vec![1, 2, 4, 3]);
        assert_eq!(
            output.as_slice(),
            vec![
                1., 2., 3., 1., 2., 3., 3., 4., 5., 3., 4., 5., 1., 2., 3., 1., 2., 3., 3., 4., 5.,
                3., 4., 5.
            ]
        );
    }
}
