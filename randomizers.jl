#=========================================================================
    This file should contain any functions that uses randomness.
    For instance, random permutation to perform cross validation
    and random permutation for Wilcoxon rank tests.
=========================================================================#

# Packages, using StatsBase because sample allows replacement
# while rand from Random does not replacement
using Random, StatsBase, IterTools

#=========================================================================
    The function slice_index takes in an array index, positive integer
    nsubindex, and split index into subindex sets with subindex length
    as equal as possible.
    - index::Array{Int, 1} is an array to be split into kfold subarrays.
      Ignores step if length(index) <= k
    - nsubindex::Int64 is number of subarrays to split into
=========================================================================#
function slice_index(index::Array{Int, 1}, nsubindex::Int64)
    subindex     = Dict{ Int64, Array{Int64, 1} }() # Initialise output
    len          = length(index) # Length of index
    intervalsize = Int64(floor(len//nsubindex)) # approximate step size
    rest         = len % nsubindex # number of elements not distributed
    nextitem     = 1 # current first element in index without class
    classsizes   = vcat( repeat([intervalsize + 1], rest),
                    repeat([intervalsize], nsubindex - rest) )

    # Slicing the index into subindices
    for i in 1:nsubindex
        # Define (k,v) pair for subindex
        subindex[i] = index[ nextitem:(nextitem + classsizes[i] - 1) ]
        # Updating pointer
        nextitem += classsizes[i]
    end

    return (classes = subindex, sizes = classsizes)
end

#=========================================================================
    randomise_index takes an array of indices and returns an
    three arrays of indices for training, testing, and tuning.
    -   kfold is the number of groups the indices to be split into.
        Default value is 10.
    -   response is the variable correspond to each index. Defaults to
        empty array. If provided the index set will preserve the ratio
        as accurate as possible.
    TODO/IDEA: Seed?
=========================================================================#
function randomize_index(index::Array{Int64, 1}, kfold = 10, response::Array{Int64, 1} = Array{Int64, 1}())
    len       = length(index) # Length of index
    perm      = sample(1:len, len, replace = false) # Random permutation
    randind   = index[perm] # Randomizing indices, indices may not be continuous

    if len <= kfold
        indexsets = Dict{ Int64, Array{Int64, 1} }(
            key => [ randind[key] ]
            for key in 1:len
        )
    elseif isempty(response) # Ignore ratio of the response
        indexsets = slice_index(randind, kfold).classes
    elseif len != length(response)
        # length(index) != length(response)
        throw("Length of index::Int64 and response::Any are different")
    else
        # length(index) = length(response)
        uresp     = unique(response) # unique elements in resposne
        randresp  = response[perm]   # Randomizing response with the same randomness of randindsubsetsubset
        currsizes = repeat([0], kfold) # Size

        # Initialising output variable
        indexsets   = Dict{ Int, Array{Int,1}}(
            key => Array{Int64, 1}() for key in 1:kfold
        )

        for u in uresp
            # TODO Need a start index in for u in uresp, otherwise the resulting split may be very uneven
            # Dividing by each unique element in response
            leastsigind              = findmin(currsizes)[2]
            randsubindex          = randind[randresp .== u]
            subslice, classsizes  = slice_index(randsubindex, kfold)
            for i in 0:(kfold-1) #leastind:(leastind+kfold)
                key = ((i+leastsigind) % kfold == 0) ? kfold : ((i+leastsigind) % kfold)
                indexsets[key] = vcat(indexsets[key], subslice[i+1])
                currsizes[key] += classsizes[i+1]
            end
            println("Sizes of each class currently is $(currsizes)")
        end
    end

    return indexsets
end
