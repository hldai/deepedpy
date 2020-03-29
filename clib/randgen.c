#include <assert.h>
#include <stdio.h>

void gen_rand_neg(double *w_f_start, double *w_f_end, int n_words, double total_freq, int n_neg, 
	double *uniform_rand_arr, int *output);

void gen_rand_neg(double *w_f_start, double *w_f_end, int n_words, double total_freq, int n_neg, 
	double *uniform_rand_arr, int *output) {
    int i = 0;
    for (i = 0; i < n_neg; ++i) {
        double v = uniform_rand_arr[i] * total_freq;
        if (v >= total_freq)
        {
            output[i] = n_words - 1;
            continue;
        }
        if (v <= 0)
        {
            output[i] = 0;
            continue;
        }

        int idx_left = 0, idx_right = n_words - 1;
		output[i] = -1;
        while (idx_left <= idx_right) {
            int idx_mid = (idx_left + idx_right) / 2;
			if (w_f_start[idx_mid] <= v && v <= w_f_end[idx_mid])
			{
				output[i] = idx_mid;
				break;
			} 
			else if (w_f_start[idx_mid] > v)
			{
				idx_right = idx_mid - 1;
			}
			else
			{
				idx_left = idx_mid + 1;
			}
        }
		assert(output[i] > -1 && output[i] < n_words);
    }
}
