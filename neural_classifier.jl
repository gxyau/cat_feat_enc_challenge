using Knet
include("onehot.jl")

input = onehot_encoding_nominal()
inputT = convert(Array{Float64,2}, transpose(input))
label = get_label()
N, M = size(input)

function predict(w, x)
	for i = 1:2:length(w)-2
        x = relu.(w[i]*x .+ w[i+1])
    end
    return w[end-1] * x .+ w[end]
end

loss(w, x, y_true) = nll(predict(w, x), y_true)
lossgradient = grad(loss)

function train(model, data, optim)
    for (x, y) in data
        grads = lossgradient(w, x, y)
		update!(model, grads, optim)
    end
end

N = 30000 # use small sized sample
train_idx = convert(Int64, N * 0.9)
w = [0.1f0*randn(Float64, 1024, M), zeros(Float64, 1024, 1),
		0.1f0*randn(Float64, 2, 1024), zeros(Float64, 2, 1)]
o = optimizers(w, Adam)
dtrn = minibatch(inputT[:, 1:train_idx], label[1:train_idx], 100)
dtst = minibatch(inputT[:, train_idx+1:N], label[train_idx+1:N], 100)

for epoch in 1:10
    train(w, dtrn, o)
	println((:epoch, epoch, :trn, accuracy(w, dtrn, predict),
			:tst, accuracy(w, dtst, predict)))
end
