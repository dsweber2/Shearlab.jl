# Submodule for imaging functions

#######################################
# Function to resize an array representation
# of an image to a certain number of pixeles
function resize_image(f, N)
    if size(f, 1) > size(f, 2)
		f = reshape(f, size(f, 2), size(f, 1), size(f, 3));
    end
	P = size(f, 1);
	# add 1 pixel boundary
	g = f;
	g = cat(2, g, reshape(g[:,1,:], size(g, 1), 1, size(g, 3)));
    g = cat(1, g, reshape(g[1,:,:], 1, size(g, 2), size(g, 3)));
	# interpolate
	t = linspace(1, P, N);
	ti = round.(Int64, floor.(t)) ; tj = round.(Int64, ceil.(t));
	fi = round.(Int64, t - floor.(t)); fj = 1 - fi;
	h = zeros(N, N, size(f, 3));
	for s in 1:size(f, 3)
	    h[:,:,s] = g[ti,ti,s] .* (fj * fj') + g[tj,tj,s] .* (fi * fi') + g[ti,tj,s] .* (fj * fi') + g[tj,ti,s] .* (fi * fj');
	end
	return h;
end # resize_image


#####################################
# Function that rescales date in a,b
"""
...
rescale(x::Array,a=0,b=1) rescales data in
[a,b]
...
"""
function rescale(x, a=0, b=1)
	m = minimum(x[:]);
	M = maximum(x[:]);

	if M - m < 1e-10
	    y = x;
	else
	    y = (b - a) * (x - m) / (M - m) + a;
	end
	return y;

end ## rescale

#####################################################
# Function that rescales a wavelet transform in an array A
function rescaleWav(A)
    v = maximum(abs.(A[:]));
    B = copy(A)
    if v > 0
        B = .5 + .5 * A / v;
    end
    return B;
end
