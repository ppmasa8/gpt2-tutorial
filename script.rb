require 'tokenizers'
require 'onnxruntime'
require 'numo/narray'

s = "Why do cats want to ride on the keyboard"

tokenizer = Tokenizers.from_pretrained("gpt2")

model = OnnxRuntime::Model.new("gpt2-lm-head-10.onnx")

ids = tokenizer.encode(s).ids

100.times do
    o = model.predict({ input1: [[ids]]})
    o = Numo::DFloat.cast(o["output1"][0])
    ids << o[true, -1, true].argmax
end

puts tokenizer.decode(ids)