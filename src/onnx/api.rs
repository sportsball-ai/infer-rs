use super::{sys, Error};
use std::{ffi::CStr, os::raw::c_char};

/// API is a step above sys in terms of ergonomics, but still very unsafe and not accessible
/// outside the crate.
#[derive(Clone, Copy)]
pub struct API(*const sys::OrtApi);

#[derive(thiserror::Error, Debug)]
pub enum NewAPIError {
    #[error("unsupported version")]
    UnsupportedVersion,
}

impl API {
    pub fn new() -> Result<API, NewAPIError> {
        unsafe {
            let get_api = (*sys::OrtGetApiBase())
                .GetApi
                .expect("GetApi should be available");
            let api = get_api(sys::ORT_API_VERSION);
            if api.is_null() {
                Err(NewAPIError::UnsupportedVersion)
            } else {
                Ok(API(api))
            }
        }
    }

    pub unsafe fn create_env(
        &self,
        default_logging_level: sys::OrtLoggingLevel,
        logid: *const c_char,
    ) -> Result<*mut sys::OrtEnv, Error> {
        let mut ret = std::ptr::null_mut();
        self.consume_status((*self.0).CreateEnv.expect("CreateEnv should be available")(
            default_logging_level,
            logid,
            &mut ret,
        ))?;
        Ok(ret)
    }

    pub unsafe fn release_env(&self, env: *mut sys::OrtEnv) {
        (*self.0)
            .ReleaseEnv
            .expect("ReleaseEnv should be available")(env)
    }

    pub unsafe fn get_allocator_with_default_options(
        &self,
    ) -> Result<*mut sys::OrtAllocator, Error> {
        let mut ret = std::ptr::null_mut();
        self.consume_status((*self.0)
            .GetAllocatorWithDefaultOptions
            .expect("GetAllocatorWithDefaultOptions should be available")(
            &mut ret
        ))?;
        Ok(ret)
    }

    pub unsafe fn allocator_free(
        &self,
        ptr: *mut sys::OrtAllocator,
        p: *mut ::std::os::raw::c_void,
    ) -> Result<(), Error> {
        self.consume_status((*self.0)
            .AllocatorFree
            .expect("AllocatorFree should be available")(
            ptr, p
        ))
    }

    pub unsafe fn create_session_options(&self) -> Result<*mut sys::OrtSessionOptions, Error> {
        let mut ret = std::ptr::null_mut();
        self.consume_status((*self.0)
            .CreateSessionOptions
            .expect("CreateSessionOptions should be available")(
            &mut ret
        ))?;
        Ok(ret)
    }

    pub unsafe fn create_session(
        &self,
        env: *const sys::OrtEnv,
        model_path: *const ::std::os::raw::c_char,
        options: *const sys::OrtSessionOptions,
    ) -> Result<*mut sys::OrtSession, Error> {
        let mut ret = std::ptr::null_mut();
        self.consume_status((*self.0)
            .CreateSession
            .expect("CreateSession should be available")(
            env, model_path, options, &mut ret,
        ))?;
        Ok(ret)
    }

    pub unsafe fn session_get_output_count(
        &self,
        sess: *const sys::OrtSession,
    ) -> Result<sys::size_t, Error> {
        let mut ret = 0;
        self.consume_status((*self.0)
            .SessionGetOutputCount
            .expect("SessionGetOutputCount should be available")(
            sess, &mut ret
        ))?;
        Ok(ret)
    }

    pub unsafe fn session_get_output_name(
        &self,
        sess: *const sys::OrtSession,
        index: sys::size_t,
        allocator: *mut sys::OrtAllocator,
    ) -> Result<*mut ::std::os::raw::c_char, Error> {
        let mut ret = std::ptr::null_mut();
        self.consume_status((*self.0)
            .SessionGetOutputName
            .expect("SessionGetOutputName should be available")(
            sess, index, allocator, &mut ret,
        ))?;
        Ok(ret)
    }

    pub unsafe fn run(
        &self,
        sess: *mut sys::OrtSession,
        run_options: *const sys::OrtRunOptions,
        input_names: &[*const ::std::os::raw::c_char],
        input: &[*const sys::OrtValue],
        output_names: &[*const ::std::os::raw::c_char],
    ) -> Result<Vec<*mut sys::OrtValue>, Error> {
        let mut outputs = vec![std::ptr::null_mut(); output_names.len()];
        self.consume_status((*self.0).Run.expect("Run should be available")(
            sess,
            run_options,
            input_names.as_ptr(),
            input.as_ptr(),
            input.len() as _,
            output_names.as_ptr(),
            output_names.len() as _,
            outputs.as_mut_ptr(),
        ))?;
        Ok(outputs)
    }

    pub unsafe fn release_session(&self, session: *mut sys::OrtSession) {
        (*self.0)
            .ReleaseSession
            .expect("ReleaseSession should be available")(session)
    }

    pub unsafe fn create_tensor_with_data_as_ort_value(
        &self,
        info: *const sys::OrtMemoryInfo,
        p_data: *mut ::std::os::raw::c_void,
        p_data_len: sys::size_t,
        shape: &[i64],
        type_: sys::ONNXTensorElementDataType,
    ) -> Result<*mut sys::OrtValue, Error> {
        let mut ret = std::ptr::null_mut();
        self.consume_status((*self.0)
            .CreateTensorWithDataAsOrtValue
            .expect("CreateTensorWithDataAsOrtValue should be available")(
            info,
            p_data,
            p_data_len,
            shape.as_ptr(),
            shape.len() as _,
            type_,
            &mut ret,
        ))?;
        Ok(ret)
    }

    pub unsafe fn release_value(&self, value: *mut sys::OrtValue) {
        (*self.0)
            .ReleaseValue
            .expect("ReleaseValue should be available")(value)
    }

    pub unsafe fn create_cpu_memory_info(
        &self,
        type_: sys::OrtAllocatorType,
        mem_type: sys::OrtMemType,
    ) -> Result<*mut sys::OrtMemoryInfo, Error> {
        let mut ret = std::ptr::null_mut();
        self.consume_status((*self.0)
            .CreateCpuMemoryInfo
            .expect("CreateCpuMemoryInfo should be available")(
            type_, mem_type, &mut ret,
        ))?;
        Ok(ret)
    }

    pub unsafe fn release_memory_info(&self, memory_info: *mut sys::OrtMemoryInfo) {
        (*self.0)
            .ReleaseMemoryInfo
            .expect("ReleaseMemoryInfo should be available")(memory_info)
    }

    pub unsafe fn get_tensor_type_and_shape(
        &self,
        value: *const sys::OrtValue,
    ) -> Result<*mut sys::OrtTensorTypeAndShapeInfo, Error> {
        let mut ret = std::ptr::null_mut();
        self.consume_status((*self.0)
            .GetTensorTypeAndShape
            .expect("GetTensorTypeAndShape should be available")(
            value, &mut ret
        ))?;
        Ok(ret)
    }

    pub unsafe fn get_tensor_mutable_data(
        &self,
        value: *mut sys::OrtValue,
    ) -> Result<*mut ::std::os::raw::c_void, Error> {
        let mut ret = std::ptr::null_mut();
        self.consume_status((*self.0)
            .GetTensorMutableData
            .expect("GetTensorMutableData should be available")(
            value, &mut ret
        ))?;
        Ok(ret)
    }

    pub unsafe fn get_tensor_element_type(
        &self,
        info: *const sys::OrtTensorTypeAndShapeInfo,
    ) -> Result<sys::ONNXTensorElementDataType, Error> {
        let mut ret = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        self.consume_status((*self.0)
            .GetTensorElementType
            .expect("GetTensorElementType should be available")(
            info, &mut ret
        ))?;
        Ok(ret)
    }

    pub unsafe fn get_dimensions_count(
        &self,
        info: *const sys::OrtTensorTypeAndShapeInfo,
    ) -> Result<sys::size_t, Error> {
        let mut ret = 0;
        self.consume_status((*self.0)
            .GetDimensionsCount
            .expect("GetDimensionsCount should be available")(
            info, &mut ret
        ))?;
        Ok(ret)
    }

    pub unsafe fn get_dimensions(
        &self,
        info: *const sys::OrtTensorTypeAndShapeInfo,
        dim_values: &mut [i64],
    ) -> Result<(), Error> {
        self.consume_status((*self.0)
            .GetDimensions
            .expect("GetDimensions should be available")(
            info,
            dim_values.as_mut_ptr(),
            dim_values.len() as _,
        ))
    }

    pub unsafe fn release_tensor_type_and_shape_info(
        &self,
        info: *mut sys::OrtTensorTypeAndShapeInfo,
    ) {
        (*self.0)
            .ReleaseTensorTypeAndShapeInfo
            .expect("ReleaseTensorTypeAndShapeInfo should be available")(info)
    }

    pub unsafe fn consume_status(&self, status: sys::OrtStatusPtr) -> Result<(), Error> {
        if status.is_null() {
            Ok(())
        } else {
            let code = (*self.0)
                .GetErrorCode
                .expect("GetErrorCode should be available")(status);
            let message = (*self.0)
                .GetErrorMessage
                .expect("GetErrorMessage should be available")(status);
            let message = CStr::from_ptr(message)
                .to_str()
                .unwrap_or("<malformed utf8 message>")
                .to_string();
            let err = Error { code, message };
            (*self.0)
                .ReleaseStatus
                .expect("ReleaseStatus should be available")(status);
            Err(err)
        }
    }
}
