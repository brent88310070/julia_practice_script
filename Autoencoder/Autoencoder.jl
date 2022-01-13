using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: mse
using Base: @kwdef
using CUDA
using MLDatasets
using Plots
using Flux: chunk
using Images

function getdata(args, device)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset	
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    # Reshape Data in order to flatten each image into a linear array
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Create DataLoaders (mini-batch iterators)
    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=args.batchsize)

    return train_loader, test_loader
end

function build_model(; imgsize=(28,28,1), nclasses=10)
    return Chain( Dense(prod(imgsize), 32, relu),
                  Dense(32, nclasses, relu),
                  Dense(nclasses, 32, relu),
                  Dense(32, prod(imgsize)))
end

function SSE(y::AbstractVector, ŷ::AbstractVector)
    res =  y - ŷ
    return res'*res
end

function SST(y::AbstractVector)
    z =  y .- mean(y)
    return z'*z
end

function SSE(Y::AbstractMatrix, Ŷ::AbstractMatrix)
    return sum(abs2, (Y - Ŷ), dims=2)
end

function SST(Y::AbstractMatrix)
    return sum(abs2, (Y .- mean(Y, dims=2)), dims=2)
end

function R_square(Y, Ŷ)
    return 1 - sum(SSE(Y, Ŷ))/sum(SST(Y))
end

function loss_Rsquare(data_loader, model, device)
    ls = 0.0f0
    num = 0
    R_square_mean = 0
    i = 0

    for (x, y) in data_loader
        x, y = device(x), device(y)
        ŷ = model(x)
        num +=  size(x)[end]
        i +=1
        R_square_mean += R_square(x, ŷ)
        ls += mse(ŷ, x)
    end
    
    
    return ls / num, R_square_mean / i
end

@kwdef mutable struct Args
    η::Float64 = 3e-4       # learning rate
    batchsize::Int = 256    # batch size
    epochs::Int = 50        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
end

function train(; kws...)
    args = Args(; kws...) # collect options in a struct for convenience

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    # Create test and train dataloaders
    train_loader, test_loader = getdata(args, device)

    # Construct model
    model = build_model() |> device
    ps = Flux.params(model) # model's trainable parameters
    
    ## Optimizer
    opt = ADAM(args.η)
    
    ## Training
    for epoch in 1:args.epochs
        for (x, y) in train_loader
            x, y = device(x), device(y) # transfer data to device
            gs = gradient(() -> mse(model(x), x), ps) # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end
        
        # Report on train and test
        train_loss, train_Rsquare = loss_Rsquare(train_loader, model, device)
        test_loss, test_Rsquare = loss_Rsquare(test_loader, model, device)
        println("Epoch=$epoch")
        println("  train_loss = $train_loss, train_Rsquare = $train_Rsquare")
        println("  test_loss = $test_loss, test_Rsquare = $test_Rsquare")
    end
    
    return model

end

model = train()

function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(x |> cpu, y_size), 28, :)...), (2, 1)))
end

samples = sigmoid.(model(gpu(xtest[:,1:16])))
image = convert_to_image(samples, 16)
save("output/manifold.png", image)


