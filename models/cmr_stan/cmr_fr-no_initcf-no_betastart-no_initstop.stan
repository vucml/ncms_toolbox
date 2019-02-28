functions {
    real cmr_lpmf(int[] recall_sequence, int LL, real init_fc, real beta_enc, real beta_rec, real P1, real P2, real stopScale, real stopShape, real gamma) {
        int LR = num_elements(recall_sequence);
        int num_context_units = LL + 1;
        int num_item_units = LL;
        int start_index = LL + 1;

        matrix[num_item_units, num_context_units] Mfc;
        matrix[num_context_units, num_item_units] Mcf;

        vector[num_context_units] c;
        vector[num_context_units] c_in;

        vector[num_item_units] f_in;
        vector[num_item_units] strength;
        vector[num_item_units] pRecall;

        real pStop;

        real rho;
        real P;

        vector[LR] loglik;

        real beta_start;
        real stopInit;

        beta_start = 0.0;
        stopInit = 0.0;

        loglik = rep_vector(0.0, LR);

        // Initialize association matrices
        Mfc = rep_matrix(0.0, num_item_units, num_context_units);
        Mcf = rep_matrix(0.0, num_context_units, num_item_units);

        for (i in 1:num_item_units) {
            Mfc[i, i] = init_fc;
        }

        c = rep_vector(0.0, num_context_units);
        c[start_index] = 1.0;

        // Simulate study period
        for (i in 1:LL) {
            for (j in 1:num_context_units) {
                c_in[j] = 0.0;
            }
            c_in[i] = 1.0;

            rho = sqrt(1 - beta_enc^2);

            c = rho * c + beta_enc * c_in;

            P = (P1 * exp(-P2 * (i - 1))) + 1.0;

            // Mfc += head(c_in, num_item_units) * c';
            // Mcf += (P * c) * (head(c_in, num_item_units))';
            Mfc[i] += (1.0 - init_fc) * c';
            for (j in 1:num_context_units) {
                Mcf[j,i] += P * c[j];
            }
        }

        // Simulate test period

        // Reinstatement of start context
        for (j in 1:num_context_units) {
            c_in[j] = 0.0;
        }
        c_in[start_index] = 1.0;

        rho = sqrt(1 + beta_start^2 * ((c' * c_in)^2 - 1)) - beta_start * (c' * c_in);

        c = rho * c + beta_start * c_in;

        for (i in 1:num_elements(recall_sequence)) {
            // Compute item weights
            f_in = (c' * Mcf)';

            // Stop probability
            pStop = stopInit + (1 - stopInit) * (1 - exp(-((i - 1) / stopScale)^stopShape));

            // Convert to softmax
            strength = exp(gamma * f_in);

            // Set strengths of previously recalled items to 0
            if (i > 1) {
                for (j in 1:(i - 1)) {
                    if ((recall_sequence[j] >= 1) && (recall_sequence[j] <= num_item_units) && (recall_sequence[i] != recall_sequence[j])) {
                        strength[recall_sequence[j]] = 0.0;
                    }
                }
            }

            pRecall = (1 - pStop) * strength / sum(strength);

            if (recall_sequence[i] == 0) {
                // "Halt" signal
                loglik[i] = log(pStop);
                break;
            }
            else if (recall_sequence[i] < 0) {
                // Intrusion, which is ignored for now
                loglik[i] = 0.0;
            }
            else if (recall_sequence[i] <= num_item_units) {
                loglik[i] = log(pRecall[recall_sequence[i]]);

                // Drift context
                c_in = Mfc[recall_sequence[i]]';
                c_in /= sqrt(dot_self(c_in));
                rho = sqrt(1 + beta_rec^2 * ((c' * c_in)^2 - 1)) - beta_rec * (c' * c_in);

                c = rho * c + beta_rec * c_in;
            }
            else {
                loglik[i] = 0.0;
            }
        }

        // print(loglik);

        return(sum(loglik));
    }

    int[] cmr_rng(int LL, real init_fc, real beta_enc, real beta_rec, real P1, real P2, real stopScale, real stopShape, real gamma) {
        int num_context_units = LL + 1;
        int num_item_units = LL;
        int start_index = LL + 1;

        matrix[num_item_units, num_context_units] Mfc;
        matrix[num_context_units, num_item_units] Mcf;

        vector[num_context_units] c;
        vector[num_context_units] c_in;

        vector[num_item_units] f_in;
        vector[num_item_units] strength;
        vector[num_item_units] pRecall;

        real pStop;

        real rho;
        real P;

        int recall_sequence[LL];

        real beta_start;
        real stopInit;

        beta_start = 0.0;
        stopInit = 0.0;

        for (i in 1:LL) {
            recall_sequence[i] = 0;
        }

        // Initialize association matrices
        Mfc = rep_matrix(0.0, num_item_units, num_context_units);
        Mcf = rep_matrix(0.0, num_context_units, num_item_units);

        for (i in 1:num_item_units) {
            Mfc[i, i] = init_fc;
        }

        c = rep_vector(0.0, num_context_units);
        c[start_index] = 1.0;

        // Simulate study period
        for (i in 1:LL) {
            for (j in 1:num_context_units) {
                c_in[j] = 0.0;
            }
            c_in[i] = 1.0;

            rho = sqrt(1 - beta_enc^2);

            c = rho * c + beta_enc * c_in;

            P = (P1 * exp(-P2 * (i - 1))) + 1.0;

            // Mfc += head(c_in, num_item_units) * c';
            // Mcf += (P * c) * (head(c_in, num_item_units))';
            Mfc[i] += (1.0 - init_fc) * c';
            for (j in 1:num_context_units) {
                Mcf[j,i] += P * c[j];
            }
        }

        // Simulate test period

        // Reinstatement of start context
        for (j in 1:num_context_units) {
            c_in[j] = 0.0;
        }
        c_in[start_index] = 1.0;

        rho = sqrt(1 + beta_start^2 * ((c' * c_in)^2 - 1)) - beta_start * (c' * c_in);

        c = rho * c + beta_start * c_in;

        for (i in 1:LL) {
            // Compute item weights
            f_in = (c' * Mcf)';

            // Stop probability
            pStop = stopInit + (1 - stopInit) * (1 - exp(-((i - 1) / stopScale)^stopShape));

            // Convert to softmax
            strength = exp(gamma * f_in);

            // Set strengths of previously recalled items to 0
            if (i > 1) {
                for (j in 1:(i - 1)) {
                    if ((recall_sequence[j] >= 1) && (recall_sequence[j] <= num_item_units) && (recall_sequence[i] != recall_sequence[j])) {
                        strength[recall_sequence[j]] = 0.0;
                    }
                }
            }

            pRecall = strength / sum(strength);

            if (bernoulli_rng(pStop) == 1) {
                recall_sequence[i] = 0;
                break;
            }
            else {
                recall_sequence[i] = categorical_rng(pRecall);

                // Drift context
                c_in = Mfc[recall_sequence[i]]';
                c_in /= sqrt(dot_self(c_in));
                rho = sqrt(1 + beta_rec^2 * ((c' * c_in)^2 - 1)) - beta_rec * (c' * c_in);

                c = rho * c + beta_rec * c_in;
            }
        }

        return(recall_sequence);
    }
}
data {
    int maxLR;
    int nSeq;
    int nSubj;

    int recall_sequences[nSeq, maxLR];
    int LL[nSeq];
    int subj[nSeq];
}
transformed data {
    int nSubjPars;
    int maxLL;

    nSubjPars = 8;
    maxLL = max(LL);
}
parameters {
    vector[nSubjPars] parMean;

    cholesky_factor_corr[nSubjPars] subjCorrChol;
    vector<lower=0>[nSubjPars] subjSD;
    matrix[nSubjPars, nSubj - 1] zSubj;

    // 1 real<lower=0> init_fc;
    // 2 real<lower=0, upper=1> beta_enc;
    // 3 real<lower=0, upper=1> beta_rec;
    // 4 real<lower=0, upper=1> beta_start;
    // 5 real<lower=0> P1;
    // 6 real<lower=0> P2;
    // 7 real<lower=0, upper=1> stopInit;
    // 8 real<lower=0> stopScale;
    // 9 real<lower=0> stopShape;
    // 10 real<lower=0> gamma;
}
transformed parameters {
    matrix[nSubjPars, nSubj] zSubjP;
    matrix[nSubjPars, nSubj] subjPars;

    {
        vector[nSubjPars] sumSubjPars;
        matrix[nSubjPars, nSubj] subjParsR;

        for (i in 1:nSubjPars) {
            sumSubjPars[i] = sum(zSubj[i]);
        }

        zSubjP = append_col(zSubj, -sumSubjPars);
        subjParsR = (diag_pre_multiply(subjSD, subjCorrChol) * zSubjP);

        subjPars[1] = inv_logit(parMean[1] + subjParsR[1]);
        subjPars[2] = inv_logit(parMean[2] + subjParsR[2]);
        subjPars[3] = inv_logit(parMean[3] + subjParsR[3]);
        subjPars[4] = exp(parMean[4] + subjParsR[4]);
        subjPars[5] = exp(parMean[5] + subjParsR[5]);
        subjPars[6] = exp(parMean[6] + subjParsR[6]);
        subjPars[7] = exp(parMean[7] + subjParsR[7]);
        subjPars[8] = exp(parMean[8] + subjParsR[8]);
    }
}
model {
    parMean ~ normal(0, 10);

    subjCorrChol ~ lkj_corr_cholesky(2);
    to_vector(zSubjP) ~ normal(0, 1);
    subjSD ~ exponential(0.1);

    for (i in 1:nSeq) {
        recall_sequences[i] ~ cmr(LL[i], subjPars[1, subj[i]], subjPars[2, subj[i]], subjPars[3, subj[i]], subjPars[4, subj[i]], subjPars[5, subj[i]], subjPars[6, subj[i]], subjPars[7, subj[i]], subjPars[8, subj[i]]);
    }
}
generated quantities {
    int pred_recall[nSeq, maxLL];

    for (i in 1:nSeq) {
        pred_recall[i] = cmr_rng(LL[i], subjPars[1, subj[i]], subjPars[2, subj[i]], subjPars[3, subj[i]], subjPars[4, subj[i]], subjPars[5, subj[i]], subjPars[6, subj[i]], subjPars[7, subj[i]], subjPars[8, subj[i]]);
    }
}
