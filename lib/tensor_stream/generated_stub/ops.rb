# This file has ben automatically generated by stubgen
# DO NOT EDIT
#
module TensorStream
  module OpStub

    ##
    # Returns x + y element-wise.
    #
    # This operation supports broadcasting
    #
    # Params:
    # +input_a+:: tensor X
    # +input_b+:: tensor Y
    #
    # Options:
    # +:name+:: Optional name
    def add(input_a, input_b, name: nil)

      input_a, input_b = apply_data_type_coercion(input_a, input_b)

      _op(:add, input_a, input_b, name: name)
    end



    ##
    # Returns the index with the largest value across axes of a tensor.
    #
    #
    # Params:
    # +input_a+:: tensor X (of type NUMERIC_TYPES)
    # +axis+:: Describes which axis of the input tensor to reduce across. For vectors, use axis = 0 (of type INTEGER_TYPES)
    #
    # Options:
    # +:name+:: Optional name
    # +:dimension+:: Same as axis
    # +:output_type+:: Output data type defaults to int32 default (:int32)
    def argmax(input_a, axis = nil, name: nil, dimension: nil, output_type: :int32)

      check_allowed_types(input_a, TensorStream::Ops::NUMERIC_TYPES)

      check_allowed_types(axis, TensorStream::Ops::INTEGER_TYPES)



      _op(:argmax, input_a, axis, name: name, dimension: dimension, output_type: output_type)
    end



    ##
    # Returns the index with the smallest value across axes of a tensor.
    #
    #
    # Params:
    # +input_a+:: tensor X (of type NUMERIC_TYPES)
    # +axis+:: Describes which axis of the input tensor to reduce across. For vectors, use axis = 0 (of type INTEGER_TYPES)
    #
    # Options:
    # +:name+:: Optional name
    # +:dimension+:: Same as axis
    # +:output_type+:: Output data type defaults to int32 default (:int32)
    def argmin(input_a, axis = nil, name: nil, dimension: nil, output_type: :int32)

      check_allowed_types(input_a, TensorStream::Ops::NUMERIC_TYPES)

      check_allowed_types(axis, TensorStream::Ops::INTEGER_TYPES)



      _op(:argmin, input_a, axis, name: name, dimension: dimension, output_type: output_type)
    end



    ##
    # Computes cos of input element-wise.
    #
    #
    # Params:
    # +input_a+:: tensor X (of type FLOATING_POINT_TYPES)
    #
    # Options:
    # +:name+:: Optional name
    def cos(input_a, name: nil)

      check_allowed_types(input_a, TensorStream::Ops::FLOATING_POINT_TYPES)



      _op(:cos, input_a, name: name)
    end



    ##
    # Returns x / y element-wise.
    #
    # This operation supports broadcasting
    #
    # Params:
    # +input_a+:: tensor X
    # +input_b+:: tensor Y
    #
    # Options:
    # +:name+:: Optional name
    def div(input_a, input_b, name: nil)

      input_a, input_b = apply_data_type_coercion(input_a, input_b)

      _op(:div, input_a, input_b, name: name)
    end



    ##
    # This operation creates a tensor of shape dims and fills it with value.
    #
    #
    # Params:
    # +dims+:: tensor shape
    # +value+:: scalar value to fill with
    #
    # Options:
    # +:name+:: Optional name
    def fill(dims, value, name: nil)



      _op(:fill, dims, value, name: name)
    end



    ##
    # Returns element-wise integer divistion.
    #
    # This operation supports broadcasting
    #
    # Params:
    # +input_a+:: tensor X
    # +input_b+:: tensor Y
    #
    # Options:
    # +:name+:: Optional name
    def floor_div(input_a, input_b, name: nil)

      input_a, input_b = apply_data_type_coercion(input_a, input_b)

      _op(:floor_div, input_a, input_b, name: name)
    end



    ##
    # Returns the truth value of (x > y) element-wise.
    #
    # This operation supports broadcasting
    #
    # Params:
    # +input_a+:: tensor X
    # +input_b+:: tensor Y
    #
    # Options:
    # +:name+:: Optional name
    def greater(input_a, input_b, name: nil)

      input_a, input_b = apply_data_type_coercion(input_a, input_b)

      _op(:greater, input_a, input_b, name: name)
    end



    ##
    # Returns the truth value of (x >= y) element-wise.
    #
    # This operation supports broadcasting
    #
    # Params:
    # +input_a+:: tensor X
    # +input_b+:: tensor Y
    #
    # Options:
    # +:name+:: Optional name
    def greater_equal(input_a, input_b, name: nil)

      input_a, input_b = apply_data_type_coercion(input_a, input_b)

      _op(:greater_equal, input_a, input_b, name: name)
    end



    ##
    # Returns the truth value of (x <= y) element-wise.
    #
    # This operation supports broadcasting
    #
    # Params:
    # +input_a+:: tensor X
    # +input_b+:: tensor Y
    #
    # Options:
    # +:name+:: Optional name
    def less_equal(input_a, input_b, name: nil)

      input_a, input_b = apply_data_type_coercion(input_a, input_b)

      _op(:less_equal, input_a, input_b, name: name)
    end



    ##
    # Multiplies matrix a by matrix b, producing a * b. The inputs must, following any transpositions, be tensors of rank 2 .
    #
    # This operation supports broadcasting
    #
    # Params:
    # +input_a+:: tensor X
    # +input_b+:: tensor Y
    #
    # Options:
    # +:transpose_a+:: Transpose matrix A first default (false)
    # +:transpose_b+:: Transpose matrix B first default (false)
    # +:name+:: Optional name
    def mat_mul(input_a, input_b, transpose_a: false, transpose_b: false, name: nil)

      input_a, input_b = apply_data_type_coercion(input_a, input_b)

      _op(:mat_mul, input_a, input_b, transpose_a: transpose_a, transpose_b: transpose_b, name: name)
    end


    alias_method :matmul, :mat_mul

    ##
    # Returns the max of x and y (i.e. x > y ? x : y) element-wise.
    #
    # This operation supports broadcasting
    #
    # Params:
    # +input_a+:: tensor X (of type NUMERIC_TYPES)
    # +input_b+:: tensor Y (of type NUMERIC_TYPES)
    #
    # Options:
    # +:name+:: Optional name
    def max(input_a, input_b, name: nil)

      check_allowed_types(input_a, TensorStream::Ops::NUMERIC_TYPES)

      check_allowed_types(input_b, TensorStream::Ops::NUMERIC_TYPES)

      input_a, input_b = apply_data_type_coercion(input_a, input_b)

      _op(:max, input_a, input_b, name: name)
    end



    ##
    # Returns the min of x and y (i.e. x < y ? x : y) element-wise.
    #
    # This operation supports broadcasting
    #
    # Params:
    # +input_a+:: tensor X (of type NUMERIC_TYPES)
    # +input_b+:: tensor Y (of type NUMERIC_TYPES)
    #
    # Options:
    # +:name+:: Optional name
    def min(input_a, input_b, name: nil)

      check_allowed_types(input_a, TensorStream::Ops::NUMERIC_TYPES)

      check_allowed_types(input_b, TensorStream::Ops::NUMERIC_TYPES)

      input_a, input_b = apply_data_type_coercion(input_a, input_b)

      _op(:min, input_a, input_b, name: name)
    end



    ##
    # Returns x * y element-wise.
    #
    # This operation supports broadcasting
    #
    # Params:
    # +input_a+:: tensor X
    # +input_b+:: tensor Y
    #
    # Options:
    # +:name+:: Optional name
    def mul(input_a, input_b, name: nil)

      input_a, input_b = apply_data_type_coercion(input_a, input_b)

      _op(:mul, input_a, input_b, name: name)
    end



    ##
    # Computes the power of one value to another X^Y element wise
    #
    # This operation supports broadcasting
    #
    # Params:
    # +input_a+:: tensor X
    # +input_b+:: tensor Y
    #
    # Options:
    # +:name+:: Optional name
    def pow(input_a, input_b, name: nil)

      input_a, input_b = apply_data_type_coercion(input_a, input_b)

      _op(:pow, input_a, input_b, name: name)
    end



    ##
    # Computes the product of elements across dimensions of a tensor.
    # Reduces input_tensor along the dimensions given in axis. Unless keepdims is true, the rank of the
    # tensor is reduced by 1 for each entry in axis. If keepdims is true, the reduced dimensions are
    # retained with length 1.
    # If axis has no entries, all dimensions are reduced, and a tensor with a single element is returned.
    #
    #
    # Params:
    # +input_a+:: tensor X
    # +axis+:: tensor X (of type INTEGER_TYPES)
    #
    # Options:
    # +:name+:: Optional name
    # +:keepdims+:: If true, retains reduced dimensions with length 1. default (false)
    def prod(input_a, axis = nil, name: nil, keepdims: false)

      check_allowed_types(axis, TensorStream::Ops::INTEGER_TYPES)



      input_a = TensorStream.convert_to_tensor(input_a)
      return input_a if input_a.shape.scalar?
      axis = cast_axis(input_a, axis)
      _op(:prod, input_a, axis, name: name, keepdims: keepdims)
    end


    alias_method :reduce_prod, :prod

    ##
    # Returns the rank of a tensor
    #
    #
    # Params:
    # +input+:: A tensor
    #
    # Options:
    # +:name+:: Optional name
    def rank(input, name: nil)



      input = convert_to_tensor(input)
      return cons(input.shape.ndims) if input.shape.known?
      _op(:rank, input, name: name)
    end



    ##
    # This operation returns a 1-D integer tensor representing the shape of input
    #
    #
    # Params:
    # +input+:: A tensor
    #
    # Options:
    # +:name+:: Optional name
    # +:out_type+:: Optional output type default (int32)
    def shape(input, name: nil, out_type: int32)



      return constant(shape_eval(input, out_type), dtype: out_type, name: "Shape/#{name}") if input.is_a?(Array) && !input[0].is_a?(Tensor)
      return constant(input.shape.shape, dtype: out_type, name: "Shape/#{input.name}_c") if shape_full_specified(input)
      _op(:shape, input, name: name, out_type: out_type)
    end



    ##
    # Computes sin of input element-wise.
    # <tt>y = sign(x) = -1 if x < 0; 0 if x == 0 or tf.is_nan(x); 1 if x > 0.</tt>
    # Zero is returned for NaN inputs.
    #
    #
    # Params:
    # +input_a+:: tensor X
    #
    # Options:
    # +:name+:: Optional name
    def sign(input_a, name: nil)



      _op(:sign, input_a, name: name)
    end



    ##
    # Computes sin of input element-wise.
    #
    #
    # Params:
    # +input_a+:: tensor X (of type FLOATING_POINT_TYPES)
    #
    # Options:
    # +:name+:: Optional name
    def sin(input_a, name: nil)

      check_allowed_types(input_a, TensorStream::Ops::FLOATING_POINT_TYPES)



      _op(:sin, input_a, name: name)
    end



    ##
    # Returns x - y element-wise.
    #
    # This operation supports broadcasting
    #
    # Params:
    # +input_a+:: tensor X
    # +input_b+:: tensor Y
    #
    # Options:
    # +:name+:: Optional name
    def sub(input_a, input_b, name: nil)

      input_a, input_b = apply_data_type_coercion(input_a, input_b)

      _op(:sub, input_a, input_b, name: name)
    end


    alias_method :subtract, :sub

    ##
    # Computes tan of input element-wise.
    #
    #
    # Params:
    # +input_a+:: tensor X (of type FLOATING_POINT_TYPES)
    #
    # Options:
    # +:name+:: Optional name
    def tan(input_a, name: nil)

      check_allowed_types(input_a, TensorStream::Ops::FLOATING_POINT_TYPES)



      _op(:tan, input_a, name: name)
    end



    ##
    # Computes tanh of input element-wise.
    #
    #
    # Params:
    # +input_a+:: tensor X (of type FLOATING_POINT_TYPES)
    #
    # Options:
    # +:name+:: Optional name
    def tanh(input_a, name: nil)

      check_allowed_types(input_a, TensorStream::Ops::FLOATING_POINT_TYPES)



      _op(:tanh, input_a, name: name)
    end



  end
end