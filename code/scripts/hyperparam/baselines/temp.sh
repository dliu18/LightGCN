#!/bin/bash

./scripts/hyperparam/baselines/pop_comp.sh gowalla mf 500 2048
./scripts/hyperparam/baselines/pop_comp.sh amazon-book mf 100 4096
./scripts/hyperparam/baselines/pop_comp.sh yelp2018 mf 300 2048

./scripts/hyperparam/baselines/pop_comp.sh gowalla lgn 500 2048
./scripts/hyperparam/baselines/pop_comp.sh amazon-book lgn 100 4096
./scripts/hyperparam/baselines/pop_comp.sh yelp2018 lgn 300 2048
