# Update guide_dir below
GUIDE_DIR = ../../guide

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

MDXML = $(DC_HOME)/dc
MDXML_BASE = $(DC_HOME)/dc
MDXML_QUIET = $(MDXML_BASE) 
MDXML_VERBOSE = $(MDXML_BASE) -v 
MDXML = $(MDXML_QUIET)

MDXMLDBG_BASE = $(DC_HOME)/mdxml.dbg
MDXMLDBG = $(MDXMLDBG_BASE)

FORCE: 

%.xml : %.md FORCE
	$(MDXML) $< -o $@

%.xmldbg : %.md
	$(MDXMLDBG) $< -o $@

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

