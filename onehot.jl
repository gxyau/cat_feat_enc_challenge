using DataFrames
using CSV

full_data = CSV.read("train.csv")

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

ok = onehot_vectorize(full_data[:nom_0], :nom_0)

emb = DataFrame(id=full_data[:id])

to_embed = vcat(Symbol.(:bin_, 0:4), Symbol.(:nom_, 0:8)) # Excluded nom_9

for sym in to_embed
    print("Processing " * string(sym) * "\n")
    to_append = onehot_vectorize(full_data[sym], sym)
    global emb = hcat(emb, to_append)
end

CSV.write("train_embedded.csv", emb)

summary(emb)
