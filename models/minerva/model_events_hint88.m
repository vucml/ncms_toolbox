function event = model_events_hint88()
%
%


def_fields.subject = [];
def_fields.session = [];
def_fields.trial = [];
def_fields.phase = [];
def_fields.type = '';
def_fields.itemno = [];
def_fields.category = [];
def_fields.cue_itemno = [];
def_fields.serial_position = [];
def_fields.time = [];
def_fields.duration = [];


% create a set of events for the Hintzman 1988 experiment 1 with
% lists of items drawn from a set of categories

n_subj = 1;

% check hintzman methods to see how many trials
n_trials = 1;

list_length = 200;

event_counter = 1;



for i=1:n_subj
  
  % session start event
  ev = struct();
  ev.subject = i;
  ev.session = 1;
  ev.type = 'SESS_START';
  ev.time = 0;
  event(event_counter) = propval(ev, def_fields);
  event_counter = event_counter + 1;

  for j=1:n_trials
    
    for k=1:list_length

      % presented word events
      % figure out how to give the lists similar category structure
      % as described by hintzman
      
      % assign each word an item number and a category
      
      ev = struct();
      ev.subject = i;
      ev.session = 1;
      ev.trial = j;
      ev.type = 'PRES_WORD';
      
      % figure out how to give the lists categorized items in
      % whatever order matches hint88 methods
      ev.itemno = [];

      %item_counter = item_counter + 1;
      ev.serial_position = k;
      % ev.time = time_counter;
      % time_counter = time_counter + opt.pres_dur;
      % time_counter = time_counter + opt.inter_item_interval;
      event(event_counter) = propval(ev, def_fields);
      event_counter = event_counter + 1;

      

    end
  end
end