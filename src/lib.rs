use std::path::Path;

#[cfg(all(feature = "coreml", target_os = "macos"))]
pub mod coreml;

#[cfg(feature = "onnx")]
pub mod onnx;

pub struct Environment {
    #[cfg(feature = "onnx")]
    onnx: onnx::Environment,
}

#[derive(thiserror::Error, Debug)]
pub enum NewEnvironmentError {
    #[cfg(feature = "onnx")]
    #[error(transparent)]
    ONNX(#[from] onnx::NewEnvironmentError),
}

#[derive(thiserror::Error, Debug)]
pub enum NewSessionError {
    #[cfg(feature = "onnx")]
    #[error(transparent)]
    ONNX(#[from] onnx::NewSessionError),
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    #[error(transparent)]
    CoreML(#[from] coreml::NewMLModelError),
    #[error("unsupported format")]
    UnsupportedFormat,
}

impl Environment {
    pub fn new() -> Result<Self, NewEnvironmentError> {
        Ok(Self {
            #[cfg(feature = "onnx")]
            onnx: onnx::Environment::new()?,
        })
    }

    pub fn new_session<P: AsRef<Path>>(&self, model_path: P) -> Result<Session, NewSessionError> {
        let model_path = model_path.as_ref();
        Ok(match model_path.extension().and_then(|s| s.to_str()) {
            #[cfg(feature = "onnx")]
            Some("onnx") => Session::ONNX(self.onnx.new_session(model_path)?),
            #[cfg(all(feature = "coreml", target_os = "macos"))]
            Some("mlmodel") => Session::CoreML(coreml::MLModel::new(model_path)?),
            _ => return Err(NewSessionError::UnsupportedFormat),
        })
    }
}

pub enum Session<'a> {
    #[cfg(feature = "onnx")]
    ONNX(onnx::Session<'a>),
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    CoreML(coreml::MLModel),
}

#[derive(thiserror::Error, Debug)]
pub enum SessionRunError {
    #[cfg(feature = "onnx")]
    #[error(transparent)]
    ONNXError(#[from] onnx::Error),
    #[cfg(feature = "onnx")]
    #[error(transparent)]
    ONNXRunError(#[from] onnx::SessionRunError),
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    #[error(transparent)]
    CoreML(#[from] coreml::PredictError),
}

impl<'a> Session<'a> {
    pub fn run<'r, I: IntoIterator<Item = (&'r str, InputTensor<'r>)>>(
        &self,
        inputs: I,
    ) -> Result<Vec<(&str, OutputTensor<'a>)>, SessionRunError> {
        match self {
            #[cfg(feature = "onnx")]
            Self::ONNX(sess) => {
                let inputs: Vec<_> = inputs
                    .into_iter()
                    .map(|(name, input)| {
                        Ok((
                            name,
                            sess.environment().new_tensor(input.data, input.shape)?,
                        ))
                    })
                    .collect::<Result<_, onnx::Error>>()?;
                let outputs = sess.run(&inputs)?;
                Ok(outputs
                    .into_iter()
                    .map(|(name, output)| (name, OutputTensor::ONNX(output)))
                    .collect())
            }
            #[cfg(all(feature = "coreml", target_os = "macos"))]
            Self::CoreML(model) => {
                let inputs: Vec<_> = inputs
                    .into_iter()
                    .map(|(name, input)| {
                        (
                            name,
                            coreml::InputTensor {
                                data: input.data,
                                shape: input.shape,
                            },
                        )
                    })
                    .collect();
                let outputs = model.predict(&inputs)?;
                Ok(outputs
                    .into_iter()
                    .map(|(name, output)| (name, OutputTensor::CoreML(output)))
                    .collect())
            }
        }
    }
}

pub struct InputTensor<'a> {
    pub data: &'a [f32],
    pub shape: &'a [usize],
}

pub enum OutputTensor<'a> {
    #[cfg(feature = "onnx")]
    ONNX(onnx::Tensor<'a>),
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    CoreML(coreml::OutputTensor),
}

impl<'a> OutputTensor<'a> {
    pub fn as_slice(&self) -> Option<&[f32]> {
        match self {
            #[cfg(feature = "onnx")]
            Self::ONNX(t) => t.as_slice(),
            #[cfg(all(feature = "coreml", target_os = "macos"))]
            Self::CoreML(t) => Some(t.as_slice()),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            #[cfg(feature = "onnx")]
            Self::ONNX(t) => t.shape(),
            #[cfg(all(feature = "coreml", target_os = "macos"))]
            Self::CoreML(t) => t.shape(),
        }
    }
}
