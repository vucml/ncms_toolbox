functions {
  
  real uss_lpmf(int[,] recalls, real stop_prob) {

    int LL = num_elements(recalls[1])-1;
    int recdims[2];
    int ntrials;
    int stop_code = LL+1;
    int going;
    vector[LL] strength;
    real sum_str;
    vector[LL+1] theta;
    vector[num_elements(recalls[1])] lprob;
    real out;
    
    recdims = dims(recalls);
    ntrials = recdims[1];

    out = 0;
    for (i in 1:ntrials){
      // initialize 
      for (j in 1:LL){strength[j] = 1;}
      for (j in 1:LL+1){lprob[j] = 0;}
      
      going = 1;
      // step through recall events
      for (j in 1:num_elements(recalls[i])){
	
	if (going) {
	  // create probability model
	  sum_str = sum(strength);
	  if (sum_str==0){
	    theta[LL+1] = 1;
	    for (k in 1:LL){theta[k]=0;}
	  }else{
	    theta[LL+1] = stop_prob;
	    for (k in 1:LL){
	      theta[k] = (1-stop_prob) * (strength[k] / sum_str);}
	  }
	  
	  if (recalls[i,j] != 0) {
	    //prob[i] = theta[recalls[i]];
	    lprob[j] = categorical_lpmf(recalls[i,j] | theta);
	    if (recalls[i,j] != stop_code) {
	      strength[recalls[i,j]]=0;}
	    else{going = 0;}
	  }}else{lprob[j] = 0;}
      }
      out += sum(lprob);
    }
    return out;
  }

  int[,] uss_rng(int ntrials, int LL, real stop_prob) {

    int stop_code = LL+1;
    int going;
    int out_pos;
    int this_recall;
    vector[LL] strength;
    real sum_str;
    vector[LL+1] theta;

    int seq[ntrials,LL+1];

    // initialize 
    for (i in 1:ntrials){for (j in 1:LL+1){seq[i,j] = 0;}}
    // iterate through trials
    for (i in 1:ntrials){
      
      for (j in 1:LL){strength[j] = 1;}

      out_pos = 1;
      going = 1;
      while (going) {
	// create probability model
	sum_str = sum(strength);      
	if (sum_str==0){
	  theta[LL+1] = 1;
	  for (j in 1:LL){theta[j]=0;}
	}else{
	  theta[LL+1] = stop_prob;
	  for (j in 1:LL){
	    theta[j] = (1-stop_prob) * (strength[j] / sum_str);}
	}
	// sample
	//print("THETA VALS: ", theta);
	this_recall = categorical_rng(theta);
	seq[i][out_pos] = this_recall;
	out_pos += 1;
	// check for termination and ensure no repeats
	if (this_recall==stop_code) { going = 0;}
	else {strength[this_recall]=0;}
      }
     
    }
    return seq;
  }

}

data {
  int ntrials;
  int LL;
  int recalls[ntrials,LL+1];
}

parameters {
  real<lower=0,upper=1> stop_prob;
}

//transformed parameters {}

model {
  stop_prob ~ uniform(0, 1);
  recalls ~ uss(stop_prob);
}

generated quantities {
  //int seq[ntrials,LL+1];
  //seq = uss_rng(ntrials, LL, stop_prob);
}
