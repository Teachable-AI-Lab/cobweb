#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <cmath>

// check if a vectoris close to 0
bool is_close_to_zero(const Eigen::VectorXd& vec, double epsilon);

// Function to display a progress bar with elapsed and estimated time
void displayProgressBar(int width, double progress_percentage, double seconds_elapsed);

// Custom random number generator function
double custom_rand();

// Function to repeat a string 'n' times
std::string repeat(std::string s, int n);

// Function to convert a double to a string with maximum precision
std::string doubleToString(double cnt);

// Function to compute the log-sum-exp of a vector of doubles
double logsumexp(std::vector<double> arr);

// Function to compute the log-sum-exp of two doubles
double logsumexp(double n1, double n2);

#endif // HELPER_H
