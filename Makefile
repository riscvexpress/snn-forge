SNN_FORGE_HOME:=$(CURDIR)
.DEFAULT_GOAL:=app_list

ifdef RVX_RELEASE_HOME
export RVX_UTIL_HOME=${RVX_RELEASE_HOME}/rvx_util
endif

include ./rvx_init.mh
-include ./rvx_config_path.mh

DATASET_DIR?=${SNN_FORGE_HOME}/dataset

###########
## SETUP ##
###########

setup_dev_env:
ifeq ($(OS),Windows_NT)
	setx SNN_FORGE_HOME $(CURDIR)
else
	${PYTHON3_CMD} ${RVX_UTIL_HOME}/configure_template.py -i ./rvx_setup.sh.template -o ./rvx_setup.sh -c CURDIR=$(CURDIR)
endif

# e.g. make setup_path DATASET_DIR=/data/snn-forge/dataset
setup_path:
	${PYTHON3_CMD} ${RVX_UTIL_HOME}/configure_template.py -i ./rvx_config_path.mh.template -o ./rvx_config_path.mh -c "DATASET_DIR=$(abspath ${DATASET_DIR})"
	@cat ./rvx_config_path.mh

# if ../.. is xarvis (has share_dataset), let it share its dataset with this project
DATASET_XARVIS_DIR?=$(abspath ${CURDIR}/../..)

setup_dataset:
	@if { MAKEFLAGS= ${MAKE} -q -C "${DATASET_XARVIS_DIR}" share_dataset >/dev/null 2>&1; [ $$? -ne 2 ]; }; then \
	  echo "make share_dataset in ${DATASET_XARVIS_DIR}"; \
	  MAKEFLAGS= ${MAKE} --no-print-directory -C "${DATASET_XARVIS_DIR}" share_dataset "DATASET_SHARE_DIR=${CURDIR}" || exit 1; \
	else \
	  echo "skipped: ${DATASET_XARVIS_DIR} has no share_dataset, so ${DATASET_DIR} is used"; \
	fi

.PHONY: setup_dataset

preinstall:
	@echo Checking OS type...
ifneq ($(OS),Windows_NT)
	@os_id=$$(. /etc/os-release && echo $$ID); \
	if [ "$$os_id" = "centos" ] || [ "$$os_id" = "rhel" ]; then \
		echo "Detected $$os_id -> using yum"; \
		sudo yum install -y sqlite-devel xz-devel; \
	elif [ "$$os_id" = "ubuntu" ] || [ "$$os_id" = "debian" ]; then \
		echo "Detected $$os_id -> using apt"; \
		sudo apt update && sudo apt install -y libsqlite3-dev liblzma-dev; \
	else \
		echo "Unknown OS: $$os_id"; \
		exit 1; \
	fi
endif
	@echo Preinstall is complete

#${PYTHON3_CMD} -m pip install pysqlite
#${PYTHON3_CMD} -m pip install sqlite3
#${PYTHON3_CMD} -m pip install numpy==1.26.4 # numpy must be < 2.0

install: setup_dataset install_verified

install_verified: install_verified_202607

# torch 2.8.0 wheel variant, from nvidia-smi:
#  - driver CUDA version: >=12.9 cu129, >=12.8 cu128, >=12.6 cu126, else cpu
#  - oldest GPU's compute capability: cu128/cu129 dropped sm_50..sm_70 (Maxwell/Pascal/Volta),
#    so < 7.5 falls back to cu126; < 5.0 (no wheel supports it) falls back to cpu;
#    unknown (compute_cap query unsupported or failed) falls back to cu126 to be safe
# Override with e.g. `make install TORCH_BACKEND=cpu`.
ifndef TORCH_BACKEND
TORCH_BACKEND := $(shell ver=$$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -n 1); \
	cap=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk -F. '/^[0-9]+\.[0-9]+/ {v=$$1*10+$$2; if (m=="" || v<m) m=v} END {print m}'); \
	if [ -z "$$ver" ]; then b=cpu; else \
	n=$$(( $$(echo $$ver | cut -d. -f1) * 100 + $$(echo $$ver | cut -d. -f2) )); \
	if [ $$n -ge 1209 ]; then b=cu129; elif [ $$n -ge 1208 ]; then b=cu128; elif [ $$n -ge 1206 ]; then b=cu126; else b=cpu; fi; fi; \
	if [ "$$b" != cpu ]; then \
	if [ -z "$$cap" ]; then b=cu126; \
	elif [ $$cap -lt 50 ]; then b=cpu; elif [ $$cap -lt 75 ]; then b=cu126; fi; fi; \
	echo $$b)
endif

install_verified_202607:
	@echo "torch backend: $(TORCH_BACKEND)"
	${PYTHON3_CMD} -m pip install pandas matplotlib tqdm==4.68.3 tonic==1.6.0
	${PYTHON3_CMD} -m pip install torch==2.8.0+$(TORCH_BACKEND) torchvision==0.23.0+$(TORCH_BACKEND) torchaudio==2.8.0+$(TORCH_BACKEND) --extra-index-url https://download.pytorch.org/whl/$(TORCH_BACKEND)
	${PYTHON3_CMD} -m pip install snntorch==1.0.0
	${PYTHON3_CMD} -m pip install nir==1.0.7 # NIR <-> .snn.jsonc in npx_cfg_convert.py

install_verified_202510:
	${PYTHON3_CMD} -m pip install tqdm==4.67.1 tonic==1.6.0
	${PYTHON3_CMD} -m pip install torch==2.4.1+cpu torchvision==0.19.1+cpu torchaudio==2.4.1 --extra-index-url https://download.pytorch.org/whl/cpu
	${PYTHON3_CMD} -m pip install snntorch==0.9.1

install_recent:
	${PYTHON3_CMD} -m pip install --upgrade tqdm tonic
	${PYTHON3_CMD} -m pip install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
	${PYTHON3_CMD} -m pip install --upgrade snntorch
	${PYTHON3_CMD} -m pip install --upgrade nir

install_first:
	${PYTHON3_CMD} -m pip install tqdm tonic
	${PYTHON3_CMD} -m pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
	${PYTHON3_CMD} -m pip install snntorch

RESULT_DIR=${CURDIR}/result

ifndef GPU_ID
	GPU_ID=-1
endif

#NUM_EPOCH=10
#NUM_REPEAT=1
NUM_SAMPLE=1
DATASET_LIST?=all

#########
## RAW ##
#########

train:
	@${PYTHON3_CMD} ./npx_trainer.py -cfg $(CFG_FILE) -cmd train quantize test -dataset ${DATASET_DIR} -o ${RESULT_DIR} -gpu ${GPU_ID}

test:
	@${PYTHON3_CMD} ./npx_trainer.py -cfg $(CFG_FILE) -cmd quantize test -dataset ${DATASET_DIR} -o ${RESULT_DIR} -gpu ${GPU_ID}

reset:
	@${PYTHON3_CMD} ./npx_trainer.py -cfg $(CFG_FILE) -cmd reset -dataset ${DATASET_DIR} -o ${RESULT_DIR} -gpu ${GPU_ID}
	
riscv:
	@${PYTHON3_CMD} ./npx_converter.py -cfg $(CFG_FILE) -o ${RESULT_DIR}

tv:
	@${PYTHON3_CMD} ./npx_testvector.py -cfg $(CFG_FILE) -cmd testvector -dataset ${DATASET_DIR} -s ${NUM_SAMPLE} -o ${RESULT_DIR}

sample:
	@${PYTHON3_CMD} ./npx_testvector.py -cfg $(CFG_FILE) -cmd testvector -dataset ${DATASET_DIR} -s ${NUM_SAMPLE} -o ${RESULT_DIR} --sample_only

prepare_dataset:
	@${PYTHON3_CMD} ./npx_opendataset_prepare.py -dataset ${DATASET_DIR} -i ${DATASET_LIST}

#########
## APP ##
#########

APP_CFG_DIR=${CURDIR}/app
CFG_SUFFIX=.snn.jsonc
ALL_CFG_LIST=$(wildcard ${APP_CFG_DIR}/*${CFG_SUFFIX})
APP_CFG_LIST = $(filter-out %_preprocess${CFG_SUFFIX}, $(ALL_CFG_LIST))
APP_CFG_NAME_LIST=$(patsubst %${CFG_SUFFIX},%,$(notdir ${APP_CFG_LIST}))

app_list:
	@echo ${APP_CFG_NAME_LIST}

${APP_CFG_NAME_LIST}:%: %.train %.test %.riscv

$(addsuffix .train, ${APP_CFG_NAME_LIST}):%.train:
	@make train CFG_FILE=${APP_CFG_DIR}/$(*)${CFG_SUFFIX}

$(addsuffix .test, ${APP_CFG_NAME_LIST}):%.test:
	@make test CFG_FILE=${APP_CFG_DIR}/$(*)${CFG_SUFFIX}

$(addsuffix .reset, ${APP_CFG_NAME_LIST}):%.reset:
	@make reset CFG_FILE=${APP_CFG_DIR}/$(*)${CFG_SUFFIX}

$(addsuffix .riscv, ${APP_CFG_NAME_LIST}):%.riscv:
	@make riscv CFG_FILE=${APP_CFG_DIR}/$(*)${CFG_SUFFIX}

$(addsuffix .tv, ${APP_CFG_NAME_LIST}):%.tv:
	@make tv CFG_FILE=${APP_CFG_DIR}/$(*)${CFG_SUFFIX}

$(addsuffix .sample, ${APP_CFG_NAME_LIST}):%.sample:
	@make sample CFG_FILE=${APP_CFG_DIR}/$(*)${CFG_SUFFIX}

##########
## TEST ##
##########

TEST_APP_CFG_FILE=${APP_CFG_DIR}/check_app${CFG_SUFFIX}

check0:
	@make reset CFG_FILE=${TEST_APP_CFG_FILE}

check1:
	@make train CFG_FILE=${TEST_APP_CFG_FILE}

check2:
	@make train riscv CFG_FILE=${TEST_APP_CFG_FILE}

check3:
	@make train riscv tv CFG_FILE=${TEST_APP_CFG_FILE} NUM_SAMPLE=${NUM_SAMPLE}

####################
## PREDEFINED APP ##
####################

PREDEFINED_APP_PREFIX_LIST=mnist_l2cf mnist_l2ff mnist_l3ccf mnist_l3fff mnist_l3cff mnist_l1f kmnist_l3fff fmnist_l2cf fmnist_l3cff fmnist_l3ccf cifar10_l5cccff gtsrb_l5cccff
NEURON_TYPE_LIST=ws8

gen_info:
	@echo ${PREDEFINED_APP_PREFIX_LIST}
	@echo ${NEURON_TYPE_LIST}

$(addsuffix .gen_cfg, ${PREDEFINED_APP_PREFIX_LIST}):%.gen_cfg:
	@${PYTHON3_CMD} ./npx_app_cfg_generator.py -app $(*) -neuron ${NEURON_TYPE_LIST} -dataset ${DATASET_DIR} -o ${APP_CFG_DIR}

$(addsuffix _ws8.reall, ${PREDEFINED_APP_PREFIX_LIST}):%_ws8.reall:
	@make $(*).gen_cfg NEURON_TYPE_LIST=ws8
	@make $(*)_ws8.reset
	@make $(*)_ws8.train

################
## FORWARDING ##
################

$(addsuffix .reall, ${PREDEFINED_APP_PREFIX_LIST}):%.reall: %_ws8.reall
