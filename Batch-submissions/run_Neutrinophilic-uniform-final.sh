#!/bin/bash
source /eos/user/o/ovchynni/Nu_Decoupling_Simple/.venv/bin/activate
python /eos/user/o/ovchynni/Nu_Decoupling_Simple/basicRunner_cli.py --llp-mass "$1" --llp-lifetime "$2" --llp-abundance "$3" --llp-pion-branching "$4" --llp-muon-branching "$5" --llp-two-nu-decay-e "$6" --llp-two-nu-decay-mu "$7" --llp-two-nu-decay-tau "$8" --nbins "$9" --Tstart "${10}" --output-folder "Neutrinophilic-uniform-final"
