# infer-rs

This is a no-frills crate for doing inference using ONNX and CoreML models:

```rust
let env = Environment::new()?;
let sess = env.new_session("my_model.onnx")?;

let input = array![[1., 2., 3.], [3., 4., 5.]];
let input = InputTensor {
    data: input.as_slice().unwrap(),
    shape: &[1, 1, 2, 3],
};

let outputs = sess.run(vec![("input_1", input)])?;
```

The same API works for CoreML models on macOS.
