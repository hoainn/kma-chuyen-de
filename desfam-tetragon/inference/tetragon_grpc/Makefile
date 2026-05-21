# Regenerate Python gRPC stubs from Tetragon .proto definitions.
#
# Prerequisite: vendor the .proto files matching your Tetragon cluster version.
# Check version with:
#   kubectl exec -n kube-system <tetragon-pod> -- tetragon version
#
# Then download the matching protos:
#   VER=v1.5.0   # ← replace with your version
#   mkdir -p proto/tetragon
#   curl -L https://raw.githubusercontent.com/cilium/tetragon/$$VER/api/v1/tetragon/tetragon.proto  -o proto/tetragon/tetragon.proto
#   curl -L https://raw.githubusercontent.com/cilium/tetragon/$$VER/api/v1/tetragon/events.proto    -o proto/tetragon/events.proto
#   curl -L https://raw.githubusercontent.com/cilium/tetragon/$$VER/api/v1/tetragon/sensors.proto   -o proto/tetragon/sensors.proto
#   curl -L https://raw.githubusercontent.com/cilium/tetragon/$$VER/api/v1/tetragon/stack.proto     -o proto/tetragon/stack.proto
#   curl -L https://raw.githubusercontent.com/cilium/tetragon/$$VER/api/v1/tetragon/capabilities.proto -o proto/tetragon/capabilities.proto

PROTO_DIR := proto
OUT_DIR   := .

.PHONY: stubs clean

stubs:
	python -m grpc_tools.protoc \
		-I$(PROTO_DIR) \
		--python_out=$(OUT_DIR) \
		--grpc_python_out=$(OUT_DIR) \
		$(PROTO_DIR)/tetragon/*.proto

clean:
	rm -f tetragon/*_pb2.py tetragon/*_pb2_grpc.py
