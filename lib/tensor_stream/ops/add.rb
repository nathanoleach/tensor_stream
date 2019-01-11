TensorStream::OpMaker.define_operation :add do |op|
  op.what_it_does "Returns x + y element-wise."

  op.parameter :input_a, "tensor X"
  op.parameter :input_b, "tensor Y"

  op.parameters_must_have_same_data_type!
  op.supports_broadcasting!

  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, params|
    x, y = params
    return [grad, grad] if shapes_fully_specified_and_equal(x, y)

    sx = ts.shape(x, name: "add/shape_x")
    sy = ts.shape(y, name: "add/shape_y")
    rx, ry = _broadcast_gradient_args(sx, sy)

    [ts.reshape(ts.reduce_sum(grad, rx, name: "add/reduce_sum_x"), sx),
     ts.reshape(ts.reduce_sum(grad, ry, name: "add/reduce_sum_y"), sy),]
  end
end