
CXX=dpcpp

gpp_obj = main.o

CXXFLAGS = -Ofast -std=c++20 -qopenmp -fsycl-targets=spir64_gen -Xs "-device xe_hp_sdv"
LDFLAGS = 

all: main 
EXEC= main

main: $(gpp_obj)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(gpp_obj) -o $(EXEC).exe

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


clean:
	rm -f *.o $(EXEC).exe
