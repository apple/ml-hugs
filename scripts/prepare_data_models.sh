#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

#!/bin/bash

# download neuman data
wget neuman_data.zip

# download pretrained models 
wget hugs_pretrained_models.zip

unzip -qq neuman_data.zip
unzip -qq hugs_pretrained_models.zip

rm neuman_data.zip
rm hugs_pretrained_models.zip