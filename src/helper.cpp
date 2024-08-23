#include "helper.h"

std::random_device rd;
std::mt19937_64 gen(rd());
std::uniform_real_distribution<double> unif(0, 1);


bool is_close_to_zero(const Eigen::VectorXd& vec, double epsilon) {
    for (int i = 0; i < vec.size(); ++i) {
        if (std::abs(vec[i]) > epsilon) {
            return false; // If any value is greater than epsilon, return false
        }
    }
    return true; // All values are within the epsilon threshold of zero
}

void displayProgressBar(int width, double progress_percentage, double seconds_elapsed)
{

    int hours = seconds_elapsed / 3600;
    int minutes = (seconds_elapsed - hours * 3600) / 60;
    int secs = seconds_elapsed - hours * 3600 - minutes * 60;

    double estimated = seconds_elapsed / progress_percentage * (1.0 - progress_percentage);

    int hours_left = estimated / 3600;
    int minutes_left = (estimated - hours_left * 3600) / 60;
    int secs_left = estimated - hours_left * 3600 - minutes_left * 60;

    int pos = width * progress_percentage;
    std::cout << "[";
    for (int i = 0; i < width; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress_percentage * 100.0) << " %; " << hours << ":" << std::setfill('0') << std::setw(2) << minutes << ":" << std::setfill('0') << std::setw(2) << secs << " elapsed; " << hours_left << ":" << std::setfill('0') << std::setw(2) << minutes_left << ":" << std::setfill('0') << std::setw(2) << secs_left << " left\r";
    std::cout.flush();
}

double custom_rand()
{
    return unif(gen);
}

std::string repeat(std::string s, int n)
{
    std::string res = "";
    for (int i = 0; i < n; i++)
    {
        res += s;
    }
    return res;
}

std::string doubleToString(double cnt)
{
    std::ostringstream stream;
    // Set stream to output floating point numbers with maximum precision
    stream << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10) << cnt;
    return stream.str();
}

double logsumexp(std::vector<double> arr)
{
    if (arr.size() > 0)
    {
        double max_val = arr[0];
        double sum = 0;

        for (auto &v : arr)
        {
            if (v > max_val)
            {
                max_val = v;
            }
        }

        for (auto &v : arr)
        {
            sum += exp(v - max_val);
        }
        return log(sum) + max_val;
    }
    else
    {
        return 0.0;
    }
}

double eff_logsumexp(double n1, double n2)
{
    if (n1 == n2) {
        return n1 + log(2.0);
    }

    double max_val = std::max(n1, n2);
    double min_val = std::min(n1, n2);

    return log1p(exp(min_val - max_val)) + max_val;
}

double logsumexp(double n1, double n2)
{
    double max_val = std::max(n1, n2);
    return log(exp(n1 - max_val) + exp(n2 - max_val)) + max_val;
}
