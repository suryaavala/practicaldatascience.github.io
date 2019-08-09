# Update guide_dir below
GUIDE_DIR = ../../guide

FLAG_VERBOSE = -v 

ifeq ($(OS),Windows_NT)
	DC_HOME = $(GUIDE_DIR)/bin/windows
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		DC_HOME = $(GUIDE_DIR)/bin/ubuntu
	endif
	ifeq ($(UNAME_S),Darwin)
		DC_HOME = $(GUIDE_DIR)/bin/macos
	endif
endif

DC_HOME = ~/DC

DC = $(DC_HOME)/dc 

FORCE: 

%.xml : %.md FORCE
ifdef debug
ifdef verbose
	$(DC) $(FLAG_VERBOSE) $< -preamble $(PREAMBLE) -o $@
 else
	$(DC) $< -preamble $(PREAMBLE) -o $@
endif
else
ifdef verbose
	$(DC) $(FLAG_VERBOSE) $< -preamble $(PREAMBLE) -o $@
 else
	$(DC) $< -preamble $(PREAMBLE) -o $@
endif
endif

clean: 
	rm *.aux *.idx *.log *.out *.toc */*.aux */*.idx */*.log */*.out 

reset: 
	make clean; rm *.pdf; rm*.html; rm  *~; rm */*~; rm  \#*\#; rm */\#*\#; 

all-xml: data_collection.xml		mle.xml \
  debugging.xml			nonlinear_modeling.xml \
  free_text.xml			probability.xml \
  graphs.xml			recommender.xml \
  intro.xml			relational_data.xml \
  linear_classification.xml	unsupervised.xml \
  matrices.xml			visualization.xml \
  ml_intro.xml 


some-xml: data_collection.xml		mle.xml \
  free_text.xml			probability.xml \
  intro.xml			relational_data.xml \
  matrices.xml			visualization.xml \
  ml_intro.xml 

