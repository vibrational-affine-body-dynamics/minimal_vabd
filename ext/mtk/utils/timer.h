#pragma once
#include <chrono>
#include <cstdio>
class Timer
{
public:
    std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
    std::chrono::time_point<std::chrono::high_resolution_clock> mEnd;
    double time;
    void start(void)
    {
        mStart = std::chrono::high_resolution_clock::now();
    }

    void end(const std::string &msg)
    {
        mEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = mEnd - mStart;
        // std::cout << msg << " took: " << duration.count() << " ms" << std::endl;
        time = duration.count(); 
        printf("%s took %.3f ms\n", msg.c_str(), time);
    }
};
