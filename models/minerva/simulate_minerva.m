function [recog_array] = simulate_minerva(param, events)
%
%

% currently expecting 1 subject's worth of events
recog_array = [];
count = 1;

for i = 1:length(events)  
  switch events(i).type
    case 'SESS_START'
      
      % this function creates the study patterns and puts them on
      % the env structure, and it creates the network structure
      % [event, env, net] = initialize_minerva(event(i), env, param);
      
      memstack = [];
      [scramlist,prototypes] = wordlist(param);
      orig_scramlist = scramlist;
      [listlength,~] = size(orig_scramlist);
      pres_order = orig_scramlist(randperm(listlength),:);
      lure_order = orig_scramlist(randperm(listlength),:);
    
    case 'PRES_WORD'
      
      % inside here, the event will have an itemno, and the env
      % will have a pattern for that itemno, so you can add that
      % item representation to the memory stack stored on the net
      % structure 
      % net = present_item_minerva(event(i), env, net, param);
      % ^ What's "net"?
      
      oldstack = memstack;
      [memstack,newscramlist] = pres_word(param,oldstack,events(i),scramlist);
      scramlist = newscramlist;
      
     
      % The following two cases need debugging
      case 'PRES_STUDIED_WORD'
          probe = pres_order(1,:);
          pres_order = pres_order(2:listlength,:);
          activations = calc_activations(param,probe,memstack);
          intensity = calc_intensity(activations);
          % make some intensity threshold for recognition.
          if intensity >= 1
              recog_array(count) = true;
          else
              recog_array(count) = false;
          end
          count = count + 1;
          
      case 'PRES_LURE'
          rand_cat = randsample(param.cpf*4,1);
          probe = prototypes(rand_cat,:);
          changeindexes = randsample(length(probe),param.distance);
          for j = 1:length(changeindexes)
              probe(changeindexes(j)) = -probe(changeindexes(j));
          end
          activations = calc_activations(param,probe,memstack);
          intensity = calc_intensity(activations);
          % make some intensity threshold for recognition.
          if intensity >= 1
              recog_array(count) = true;
          else
              recog_array(count) = false;
          end
          count = count + 1;
        
  end
      
  
  
end






