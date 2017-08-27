default: build
deps:
	go get github.com/pkg/profile
build:
	go build -ldflags="-s -w"
linux:
	GOOS=linux GOARCH=amd64 go build -ldflags="-s -w"
