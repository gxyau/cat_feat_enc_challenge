using DataFrames
using CSV

function onehot_vectorize(col, sym)
    lv = length(unique(col))
    colnames = Symbol.(unique(col))
    df = DataFrame(repeat([Float64], lv), colnames, length(col))
    for c in unique(col)
        df[Symbol(c)] = Float64.(col .== c)
    end
    names!(df, Symbol.(sym, "_", colnames))
    return df
end

function onehot_encoding_nominal()
    full_data = CSV.read("train.csv")
    emb = DataFrame(id=full_data[:id])
    to_embed = vcat(Symbol.(:bin_, 0:4), Symbol.(:nom_, 0:8)) # Excluded nom_9
    for sym in to_embed
        print("Processing " * string(sym) * "\n")
        to_append = onehot_vectorize(full_data[sym], sym)
        emb = hcat(emb, to_append)
    end
    nominal_data = convert(Array, emb)
    nominal_data = nominal_data[:, 2:length(nominal_data[1, :])]
    return nominal_data
end

function get_label() # added 1 since Julia is not using 0 index...
    full_data = CSV.read("train.csv")
	return full_data[:, :target] .+ 1
end
