#=========================================================================
    randomise_index takes an array of indices and returns an
    three arrays of indices for training, testing, and tuning.
    -   kfold is the number of groups the indices to be split into.
        Default value is 10.
    -   response is the variable correspond to each index. Defaults to
        empty array. If ratio is set to true, an array of equal length
        to index should be provided
    -   ratio is a flag on whether or not the ratio of the response
        should be taken into account. Defaults to false. If set to
        true, then response cannot be empty.
=========================================================================#
# Packages, using StatsBase because sample allows replacement
# while rand from Random does not replacement
using StatsBase

#=
    TODO: This is wasting resources, should return k groups instead
    IDEA: Dynamically name the groups ind1, ..., indk and return (ind1,..., indk)
=#
function randomise_index(index::Array{Int64, 1}, kfold = 10, response = [], ratio = false)
    # Initialising output indices
    train, test, tune = Array{Int64, 1}(), Array{Int64, 1}(), Array{Int64, 1}()

    # Sampling
    len = length(index) # Length of index
    if !(ratio)
        # ratio = false
        stepsize = Int64(floor(len//k)) # len//k is the fraction
        randind  = sample(index, len, replace = false)
        test     = randind[1:stepsize]
        tune     = randind[(stepsize+1) : (2*stepsize)]
        train    = randind[(2*stepsize+1) : len]
    else if response = []
        # ratio = true, length(response) = 0
        throw("response::Any cannot be empty when ratio = true")
    else if len != length(response)
        # ratio = true, length(index) != length(response)
        throw("Length of index::Int64 and response::Any are different")
    else
        # ratio = true, length(index) = length(response)
        uniqueresp = unique(response) # unique elements in resposne
        for u in uniqueresp
            # Dividing by each unique element in response
            subindex   = index[response .== u]
            sublen     = length(subindex)
            stepsizeu  = Int64(floor(sublen//k))
            randindu   = sample(subindex, sublen, replace = false)
            test       = vcat(test, randindu[1:stepsizeu])
            tune       = vcat(tune, randindu[(stepsizeu+1) : (2*stepsizeu)])
            train      = vcat(train, randindu[(2*stepsizeu+1) : sublen])
        end
    end

    return (train = train, tune = tune, test = test)
end
