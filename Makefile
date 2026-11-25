# Builds RECOUP and its experiment variants (C++17, zero external dependencies).
CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17
DIR      := harness/cpp/dylo
VARIANTS := std4 std4sel std4w seltime eps seedrep b3fix r2greedy
BINS     := $(addprefix $(DIR)/dylo_driver_,$(VARIANTS)) $(DIR)/selector_verify $(DIR)/hit_leiden_driver_inc
ALIAS    := $(DIR)/dylo_driver

all: $(BINS) $(ALIAS)

$(DIR)/dylo_driver_%: $(DIR)/driver_%.cxx $(wildcard $(DIR)/*.hpp)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(ALIAS): $(DIR)/dylo_driver_std4
	ln -sf dylo_driver_std4 $@

$(DIR)/selector_verify: $(DIR)/selector_verify.cxx $(wildcard $(DIR)/*.hpp)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(DIR)/hit_leiden_driver_inc: $(DIR)/hit_leiden_driver_inc.cxx $(DIR)/detector.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

verify:
	python3 scripts/verify_paper_tables.py
	python3 scripts/cert_oracle_test.py

smoke: $(DIR)/dylo_driver_std4
	bash scripts/smoke_test.sh

figures:
	mkdir -p results/figures_regen
	python3 scripts/plot_fig2_motivation.py results/figures_regen
	python3 scripts/plot_fig5_fig6.py results/figures_regen

clean:
	rm -f $(BINS) $(ALIAS)

.PHONY: all verify smoke figures clean
