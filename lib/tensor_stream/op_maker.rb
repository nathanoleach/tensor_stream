class TensorStream::OpMaker
  attr_reader :operation, :description, :parameters,
              :options, :gradient, :check_types,
              :supports_broadcast, :data_type_coercion

  def initialize(op)
    @operation = op
    @parameters = []
    @options = {}
    @gradient = nil
    @supports_broadcast = false
    @data_type_coercion = false
  end

  def self.scan
    op_files = Dir[File.join("lib", "tensor_stream", "ops", "*.rb")]
    op_files.each { |file|
      load File.join("tensor_stream", "ops", File.basename(file))
    }
  end

  def self.define_operation(op_code, &block)
    @ops ||= {}
    op_maker = TensorStream::OpMaker.new(op_code.to_sym)
    block.call(op_maker)
    @ops[op_code.to_sym] = op_maker
  end

  # call an operations' gradient definition
  def self.gradient_op(context_caller, node, grad)
    raise "No derivative op defined for #{node.operation}" if @ops[node.operation].nil? || @ops[node.operation].gradient.nil?

    context_caller.instance_exec(grad, node, node.inputs, &@ops[node.operation].gradient)
  end

  def self.each_op(&block)
    @ops.values.sort_by { |op| op.operation }.each do |op|
      block.call(op)
    end
  end

  def what_it_does(description)
    @description = description
  end

  ##
  # adds a parameter to the op
  #
  def parameter(name, description, default_value = nil, validate: nil)
    @parameters << {
      name: name.to_s,
      description: description,
      default_value: default_value,
      validate: validate
    }
  end

  def option(name, description, default_value = nil)
    @options[name] = { description: description, default_value: default_value }
  end

  def define_gradient(&block)
    @gradient = block
  end

  def expand_params(print_defaults)
    @parameters.map { |param|
      print_defaults && param[:default_value] ? "#{param[:name]} = #{default_with_nil(param[:default_value])}" : "#{param[:name]}"
    }
  end

  def parameters_must_have_same_data_type!
    @check_types = true
  end

  def apply_data_type_coercion!
    @data_type_coercion = true
  end

  def supports_broadcasting!
    if (@parameters.size> 1)
      @supports_broadcast = true
    else
      raise "Ops with parameters < 2 cannot support broadcasting"
    end
  end

  def supports_broadcasting?
    @supports_broadcast
  end

  def data_type_coercion?
    @data_type_coercion
  end

  def check_types?
    @check_types
  end

  def expand_options(print_defaults)
    @options.map { |k, v|
      print_defaults && v[:default_value] ? "#{k}: #{default_with_nil(v[:default_value])}" : "#{k}:"
    }
  end

  def options_call
    @options.map { |k, v|
      "#{k}: #{k}"
    }
  end

  def default_with_nil(v)
    v == :nil ? 'nil' : v
  end
end

TensorStream::OpMaker.scan
