BLUE := \033[36m
BOLD := \033[1m
RESET := \033[0m

.DEFAULT_GOAL := help

.PHONY: build
build: ## build the library
	@printf "$(BLUE)Building cvxrust...$(RESET)\n"
	@cargo build

.PHONY: test
test: ## run tests with all features, including doctests
	@printf "$(BLUE)Running tests with all features...$(RESET)\n"
	@cargo test --all-features

.PHONY: examples
examples: ## build and run all examples
	@printf "$(BLUE)Building and running examples...$(RESET)\n"
	@cargo build --examples
	@for example in examples/*.rs; do \
		cargo run --example "$$(basename "$$example" .rs)" || exit $$?; \
	done

.PHONY: fmt
fmt: ## format all Rust source files
	@printf "$(BLUE)Formatting Rust sources...$(RESET)\n"
	@cargo fmt --all

.PHONY: lint
lint: ## check formatting and lint all targets and features
	@printf "$(BLUE)Checking Rust formatting...$(RESET)\n"
	@cargo fmt --all -- --check
	@printf "$(BLUE)Linting cvxrust...$(RESET)\n"
	@cargo clippy --all-targets --all-features -- -D warnings

.PHONY: docs
docs: ## generate the API reference without dependency documentation
	@printf "$(BLUE)Building documentation...$(RESET)\n"
	@cargo doc --no-deps --all-features

.PHONY: preview
preview: ## build and open the API reference in a browser
	@cargo doc --no-deps --all-features --open

.PHONY: package
package: ## verify packaging with a publishing dry run
	@printf "$(BLUE)Verifying package...$(RESET)\n"
	@cargo publish --dry-run

.PHONY: clean
clean: ## remove Cargo build artifacts
	@printf "$(BLUE)Cleaning Cargo build artifacts...$(RESET)\n"
	@cargo clean

.PHONY: check
check: lint test examples docs ## run formatting, lint, test, example, and documentation checks

.PHONY: help
help: ## display this help message
	@printf "$(BOLD)Usage:$(RESET)\n"
	@printf "  make $(BLUE)<target>$(RESET)\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  $(BLUE)%-25s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
