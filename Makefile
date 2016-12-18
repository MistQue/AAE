CC = g++ -std=c++11
LNK = `pkg-config opencv --cflags --libs`

all:test.cpp
	$(CC) test.cpp $(LNK) 
clean:
	rm -rf a.out
