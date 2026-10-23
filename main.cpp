#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <string>
#include <random>
#include <deque>
#include <unordered_map>
#include "json.hpp"

using json = nlohmann::json;

// --- Fundamental Physical Constants (cgs / Gaussian) ---
constexpr double C_SPEED = 2.99792458e10;
constexpr double MASS = 9.1093837015e-28;
constexpr double CHARGE = 4.80320471e-10;
constexpr double H_BAR = 1.054571817e-27;
constexpr double BOHR_RADIUS = 5.291772109e-9;
constexpr double TAU = (2.0 * CHARGE * CHARGE) / (3.0 * MASS * C_SPEED * C_SPEED * C_SPEED);

/************************************************************************************************
 1. The Core Data Structures & Memory
 ************************************************************************************************/
struct VacuumBin {
    double Ex_A, Ex_B;
    double Ey_A, Ey_B;
    double Ez_A, Ez_B;
    
    double Bx_A, Bx_B;
    double By_A, By_B;
    double Bz_A, Bz_B;
};

std::unordered_map<int, VacuumBin> vacuum_cache;
std::vector<double> universal_log_w;
std::vector<double> universal_log_dw;

std::mt19937 gen;

const double ZPF_CONSTANT = (2.0 * H_BAR) / (3.0 * M_PI * C_SPEED * C_SPEED * C_SPEED);

// --- Global Diagnostic & Tracking Variables ---
double current_a;
double current_e;
double current_theta;
double current_w_c;

// --- Configuration Variables ---
double window_spread;
double taper_width;
double a_inner;
double a_outer;
int max_stride;

/*******************************************************************************************
 2. The Universal Log Grid Initialization
 ********************************************************************************************/
void initializeUniversalGrid(double w_min_absolute, double w_max_absolute, int total_bins) {
    universal_log_w.resize(total_bins);
    universal_log_dw.resize(total_bins);
    
    double log_step = std::log(w_max_absolute / w_min_absolute) / (total_bins - 1);
    for (int i = 0; i < total_bins; ++i) {
        universal_log_w[i] = w_min_absolute * std::exp(i * log_step);
    }
    for (int i = 0; i < total_bins - 1; ++i) {
        universal_log_dw[i] = universal_log_w[i+1] - universal_log_w[i];
    }
    universal_log_dw[total_bins - 1] = universal_log_dw[total_bins - 2];
}

/*************************************************************************************************
 3. The Lazy Generator (The Physics Engine)
 **************************************************************************************************/
VacuumBin getOrGenerateVacuumBin(int index) {
    if (vacuum_cache.find(index) == vacuum_cache.end()) {
        std::normal_distribution<double> d(0.0, 1.0);
        VacuumBin new_bin;
        
        new_bin.Ex_A = d(gen); new_bin.Ex_B = d(gen);
        new_bin.Ey_A = d(gen); new_bin.Ey_B = d(gen);
        new_bin.Ez_A = d(gen); new_bin.Ez_B = d(gen);
        
        new_bin.Bx_A = d(gen); new_bin.Bx_B = d(gen);
        new_bin.By_A = d(gen); new_bin.By_B = d(gen);
        new_bin.Bz_A = d(gen); new_bin.Bz_B = d(gen);
        
        vacuum_cache[index] = new_bin;
    }
    return vacuum_cache[index];
}

/**************************************************************************************************
 4. The Angular Ring Buffer (The Stable Anchor)
 ***************************************************************************************************/
struct StatePoint {
    double x, y, z;
    double time;
};

const int FIT_POINTS = 200;
StatePoint orbit_history[FIT_POINTS];
int history_index = 0;

const double ANGULAR_THRESHOLD = 0.087;
double last_recorded_angle = 0.0;

bool updateRingBuffer(double current_x, double current_y, double current_z, double current_t, double current_angle) {
    if (std::abs(current_angle - last_recorded_angle) >= ANGULAR_THRESHOLD) {
        orbit_history[history_index] = {current_x, current_y, current_z, current_t};
        history_index = (history_index + 1) % FIT_POINTS;
        last_recorded_angle = current_angle;
        
        if (history_index == 0) return true;
    }
    return false;
}

// --- The Tukey Smoothing Window ---
double calculateTukeyWeight(double omega, double w_min, double w_max, double r_taper) {
    if (omega <= w_min || omega >= w_max) return 0.0;
    
    double width = w_max - w_min;
    double x = (omega - w_min) / width;
    
    if (x < r_taper / 2.0) {
        return 0.5 * (1.0 - std::cos((2.0 * M_PI * x) / r_taper));
    }
    else if (x > 1.0 - (r_taper / 2.0)) {
        return 0.5 * (1.0 - std::cos((2.0 * M_PI * (1.0 - x)) / r_taper));
    }
    return 1.0;
}

/************************************************************************************************
 5. The Active Window Evaluation
 *************************************************************************************************/
void computeZPFForce(double current_time,
                     double& total_Ex, double& total_Ey, double& total_Ez,
                     double& total_Bx, double& total_By, double& total_Bz,
                     double& total_dEx, double& total_dEy, double& total_dEz,
                     double& total_dBx, double& total_dBy, double& total_dBz) {
    
    double fractional_depth = (current_a - a_inner) / (a_outer - a_inner);
    fractional_depth = std::max(0.0, std::min(1.0, fractional_depth));
    
    int current_stride = std::max(1, static_cast<int>(max_stride * fractional_depth));
    
    double w_min_active = current_w_c * (1.0 - window_spread);
    double w_max_active = current_w_c * (1.0 + window_spread);
    
    total_Ex = 0; total_Ey = 0; total_Ez = 0;
    total_Bx = 0; total_By = 0; total_Bz = 0;
    total_dEx = 0; total_dEy = 0; total_dEz = 0;
    total_dBx = 0; total_dBy = 0; total_dBz = 0;
    
    auto it_start = std::lower_bound(universal_log_w.begin(), universal_log_w.end(), w_min_active);
    auto it_end = std::upper_bound(it_start, universal_log_w.end(), w_max_active);

    int start_idx = std::distance(universal_log_w.begin(), it_start);
    int end_idx = std::distance(universal_log_w.begin(), it_end);
    
    for (int i = start_idx; i < end_idx; i += current_stride) {
        VacuumBin bin = getOrGenerateVacuumBin(i);
        double w = universal_log_w[i];
                
        double effective_dw = universal_log_dw[i] * current_stride;
        double shell_variance = ZPF_CONSTANT * std::pow(w, 3) * effective_dw;
        double amplitude_scale = std::sqrt(shell_variance);

        double tukey_weight = calculateTukeyWeight(w, w_min_active, w_max_active, taper_width);
        amplitude_scale *= tukey_weight;

        double phase = w * current_time;
        double cos_wt = std::cos(phase);
        double sin_wt = std::sin(phase);
        
        total_Ex += amplitude_scale * (bin.Ex_A * cos_wt - bin.Ex_B * sin_wt);
        total_Ey += amplitude_scale * (bin.Ey_A * cos_wt - bin.Ey_B * sin_wt);
        total_Ez += amplitude_scale * (bin.Ez_A * cos_wt - bin.Ez_B * sin_wt);
        
        total_Bx += amplitude_scale * (bin.Bx_A * cos_wt - bin.Bx_B * sin_wt);
        total_By += amplitude_scale * (bin.By_A * cos_wt - bin.By_B * sin_wt);
        total_Bz += amplitude_scale * (bin.Bz_A * cos_wt - bin.Bz_B * sin_wt);
        
        total_dEx += amplitude_scale * w * (-bin.Ex_A * sin_wt - bin.Ex_B * cos_wt);
        total_dEy += amplitude_scale * w * (-bin.Ey_A * sin_wt - bin.Ey_B * cos_wt);
        total_dEz += amplitude_scale * w * (-bin.Ez_A * sin_wt - bin.Ez_B * cos_wt);
        
        total_dBx += amplitude_scale * w * (-bin.Bx_A * sin_wt - bin.Bx_B * cos_wt);
        total_dBy += amplitude_scale * w * (-bin.By_A * sin_wt - bin.By_B * cos_wt);
        total_dBz += amplitude_scale * w * (-bin.Bz_A * sin_wt - bin.Bz_B * cos_wt);
    }
}

/************************************************************************************************
 6. The 3x3 Spatial Plane Fitter
 *************************************************************************************************/
void calculateOrbitalPlaneNormal(const StatePoint history[], int num_points,
                                double& nx, double& ny, double& nz,
                                double& mean_x, double& mean_y, double& mean_z) {
    mean_x = 0; mean_y = 0; mean_z = 0;
    for (int i = 0; i < num_points; ++i) {
        mean_x += history[i].x; mean_y += history[i].y; mean_z += history[i].z;
    }
    mean_x /= num_points; mean_y /= num_points; mean_z /= num_points;

    double C[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
    for (int i = 0; i < num_points; ++i) {
        double dx = history[i].x - mean_x;
        double dy = history[i].y - mean_y;
        double dz = history[i].z - mean_z;
        C[0][0] += dx * dx; C[0][1] += dx * dy; C[0][2] += dx * dz;
                            C[1][1] += dy * dy; C[1][2] += dy * dz;
                                                C[2][2] += dz * dz;
    }
    C[1][0] = C[0][1]; C[2][0] = C[0][2]; C[2][1] = C[1][2];

    double V[3][3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    for (int sweep = 0; sweep < 50; ++sweep) {
        double max_off_diag = 0.0;
        int p = 0, q = 1;
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                if (std::abs(C[i][j]) > max_off_diag) {
                    max_off_diag = std::abs(C[i][j]);
                    p = i; q = j;
                }
            }
        }
        
        double trace = C[0][0] + C[1][1] + C[2][2];
        if (max_off_diag < 1e-14 * trace || max_off_diag == 0.0) break;
        
        double theta = 0.5 * std::atan2(2.0 * C[p][q], C[q][q] - C[p][p]);
        double cos_t = std::cos(theta); double sin_t = std::sin(theta);
        
        double C_pp = cos_t * cos_t * C[p][p] - 2.0 * sin_t * cos_t * C[p][q] + sin_t * sin_t * C[q][q];
        double C_qq = sin_t * sin_t * C[p][p] + 2.0 * sin_t * cos_t * C[p][q] + cos_t * cos_t * C[q][q];
        C[p][q] = C[q][p] = 0.0; C[p][p] = C_pp; C[q][q] = C_qq;
        
        for (int i = 0; i < 3; ++i) {
            if (i != p && i != q) {
                double C_ip = C[i][p]; double C_iq = C[i][q];
                C[i][p] = C[p][i] = cos_t * C_ip - sin_t * C_iq;
                C[i][q] = C[q][i] = sin_t * C_ip + cos_t * C_iq;
            }
        }
        for (int i = 0; i < 3; ++i) {
            double V_ip = V[i][p]; double V_iq = V[i][q];
            V[i][p] = cos_t * V_ip - sin_t * V_iq;
            V[i][q] = sin_t * V_ip + cos_t * V_iq;
        }
    }

    int min_index = 0; double min_eigenvalue = C[0][0];
    if (C[1][1] < min_eigenvalue) { min_eigenvalue = C[1][1]; min_index = 1; }
    if (C[2][2] < min_eigenvalue) { min_eigenvalue = C[2][2]; min_index = 2; }

    nx = V[0][min_index]; ny = V[1][min_index]; nz = V[2][min_index];
    if (nz < 0) { nx = -nx; ny = -ny; nz = -nz; }
}

/************************************************************************************************
 7. Simulation Config & State Structs
 *************************************************************************************************/
struct State {
    double x, y, z, vx, vy, vz;
};

struct AppliedWave {
    std::string type;
    double amplitude;
    double frequency;
    double phase;
    int rotation_direction;
};

struct SimConfig {
    unsigned int rng_seed;
    std::vector<AppliedWave> applied_waves;
    
    int print_frequency;
    
    double dt_max = 5.0e-19;
    double dt_min = 1.0e-22;
    double tolerance = 1.0e-15;
    
    bool enable_coulomb_rr;
    bool enable_zpp_rr;
    bool enable_zp;
    bool enable_magnetic_lorentz;
};

// --- Equations of Motion ---
void computeDerivatives(double t, const State& s, State& deriv, const SimConfig& config) {
    deriv.x = s.vx; deriv.y = s.vy; deriv.z = s.vz;

    double r_sq = s.x * s.x + s.y * s.y + s.z * s.z;
    double r = std::sqrt(r_sq);
    double e_sq_over_m = (CHARGE * CHARGE) / MASS;
    double r_cubed = r * r_sq;

    double a_coulomb_x = -e_sq_over_m * (s.x / r_cubed);
    double a_coulomb_y = -e_sq_over_m * (s.y / r_cubed);
    double a_coulomb_z = -e_sq_over_m * (s.z / r_cubed);

    double r_to_the_fifth = r_cubed * r_sq;
    double x_dot_v = s.x * s.vx + s.y * s.vy + s.z * s.vz;
    double rr_prefactor = (TAU * CHARGE * CHARGE) / MASS;

    double a_rr_x = 0.0, a_rr_y = 0.0, a_rr_z = 0.0;
    if (config.enable_coulomb_rr) {
        a_rr_x = rr_prefactor * (-s.vx / r_cubed + (3.0 * s.x * x_dot_v) / r_to_the_fifth);
        a_rr_y = rr_prefactor * (-s.vy / r_cubed + (3.0 * s.y * x_dot_v) / r_to_the_fifth);
        a_rr_z = rr_prefactor * (-s.vz / r_cubed + (3.0 * s.z * x_dot_v) / r_to_the_fifth);
    }

    double a_zpf_x = 0.0, a_zpf_y = 0.0, a_zpf_z = 0.0;
    double a_zpp_rr_x = 0.0, a_zpp_rr_y = 0.0, a_zpp_rr_z = 0.0;
        
    if (config.enable_zp) {
        double Ex = 0, Ey = 0, Ez = 0;
        double Bx = 0, By = 0, Bz = 0;
        double dEx_dt = 0, dEy_dt = 0, dEz_dt = 0;
        double dBx_dt = 0, dBy_dt = 0, dBz_dt = 0;
        
        computeZPFForce(t, Ex, Ey, Ez, Bx, By, Bz, dEx_dt, dEy_dt, dEz_dt, dBx_dt, dBy_dt, dBz_dt);
        
        double e_over_m_cgs = CHARGE / MASS;
        
        a_zpf_x = -e_over_m_cgs * Ex;
        a_zpf_y = -e_over_m_cgs * Ey;
        a_zpf_z = -e_over_m_cgs * Ez;
        
        if (config.enable_magnetic_lorentz) {
            double mag_pref = -CHARGE / (MASS * C_SPEED);
            a_zpf_x += mag_pref * (s.vy * Bz - s.vz * By);
            a_zpf_y += mag_pref * (s.vz * Bx - s.vx * Bz);
            a_zpf_z += mag_pref * (s.vx * By - s.vy * Bx);
        }
        
        if (config.enable_zpp_rr) {
            a_zpp_rr_x = TAU * (-e_over_m_cgs * dEx_dt);
            a_zpp_rr_y = TAU * (-e_over_m_cgs * dEy_dt);
            a_zpp_rr_z = TAU * (-e_over_m_cgs * dEz_dt);
            
            if (config.enable_magnetic_lorentz) {
                double mag_pref = -CHARGE / (MASS * C_SPEED);
                a_zpp_rr_x += TAU * mag_pref * (s.vy * dBz_dt - s.vz * dBy_dt);
                a_zpp_rr_y += TAU * mag_pref * (s.vz * dBx_dt - s.vx * dBz_dt);
                a_zpp_rr_z += TAU * mag_pref * (s.vx * dBy_dt - s.vy * dBx_dt);
            }
        }
    }
    
    double a_app_x = 0.0, a_app_y = 0.0, a_app_z = 0.0;
    double charge_over_mass = CHARGE / MASS;
    for (const auto& wave : config.applied_waves) {
        double arg = wave.frequency * t - wave.phase;
        if (wave.type == "circular") {
            a_app_x += -charge_over_mass * wave.amplitude * std::cos(arg);
            a_app_y += -charge_over_mass * wave.rotation_direction * wave.amplitude * std::sin(arg);
        }
    }

    deriv.vx = a_coulomb_x + a_rr_x + a_zpf_x + a_app_x + a_zpp_rr_x;
    deriv.vy = a_coulomb_y + a_rr_y + a_zpf_y + a_app_y + a_zpp_rr_y;
    deriv.vz = a_coulomb_z + a_rr_z + a_zpf_z + a_app_z + a_zpp_rr_z;
}

// --- Helper: Pure RK4 Math ---
State takeRK4Step(const State& s, double t, double h, const SimConfig& config) {
    State k1, k2, k3, k4, temp;
    computeDerivatives(t, s, k1, config);

    temp = {s.x + 0.5*h*k1.x, s.y + 0.5*h*k1.y, s.z + 0.5*h*k1.z,
            s.vx + 0.5*h*k1.vx, s.vy + 0.5*h*k1.vy, s.vz + 0.5*h*k1.vz};
    computeDerivatives(t + 0.5*h, temp, k2, config);

    temp = {s.x + 0.5*h*k2.x, s.y + 0.5*h*k2.y, s.z + 0.5*h*k2.z,
            s.vx + 0.5*h*k2.vx, s.vy + 0.5*h*k2.vy, s.vz + 0.5*h*k2.vz};
    computeDerivatives(t + 0.5*h, temp, k3, config);

    temp = {s.x + h*k3.x, s.y + h*k3.y, s.z + h*k3.z,
            s.vx + h*k3.vx, s.vy + h*k3.vy, s.vz + h*k3.vz};
    computeDerivatives(t + h, temp, k4, config);

    State next_s;
    next_s.x = s.x + (h / 6.0) * (k1.x + 2.0*k2.x + 2.0*k3.x + k4.x);
    next_s.y = s.y + (h / 6.0) * (k1.y + 2.0*k2.y + 2.0*k3.y + k4.y);
    next_s.z = s.z + (h / 6.0) * (k1.z + 2.0*k2.z + 2.0*k3.z + k4.z);
    next_s.vx = s.vx + (h / 6.0) * (k1.vx + 2.0*k2.vx + 2.0*k3.vx + k4.vx);
    next_s.vy = s.vy + (h / 6.0) * (k1.vy + 2.0*k2.vy + 2.0*k3.vy + k4.vy);
    next_s.vz = s.vz + (h / 6.0) * (k1.vz + 2.0*k2.vz + 2.0*k3.vz + k4.vz);
    
    return next_s;
}

// --- The Capped Adaptive Stepper ---
bool performAdaptiveStepRK4(State& s, double& t, double& dt, const SimConfig& config) {
    const double SAFETY = 0.9;
    
    double current_dt_max = config.dt_max;
    double current_dt_min = config.dt_min;
    double current_tol = config.tolerance;

    while (true) {
        State s_full = takeRK4Step(s, t, dt, config);

        State s_half_1 = takeRK4Step(s, t, dt / 2.0, config);
        State s_half_2 = takeRK4Step(s_half_1, t + (dt / 2.0), dt / 2.0, config);

        double err_x = std::abs(s_full.x - s_half_2.x);
        double err_y = std::abs(s_full.y - s_half_2.y);
        double err_z = std::abs(s_full.z - s_half_2.z);
        double max_err = std::max({err_x, err_y, err_z});
        
        if (max_err == 0.0) max_err = 1e-30;

        if (max_err <= current_tol) {
            s = s_half_2;
            t += dt;
            
            double dt_new = dt * SAFETY * std::pow(current_tol / max_err, 0.2);
            dt = std::min(dt_new, current_dt_max);
            return true;
        } else {
            double dt_new = dt * SAFETY * std::pow(current_tol / max_err, 0.25);
            if (dt_new < current_dt_min) return false;
            dt = dt_new;
        }
    }
}

// =========================================================
// --- MAIN FUNCTION ---
// =========================================================
    
int main(int argc, char* argv[]) {
    
#ifdef __FAST_MATH__
    std::cout << "HARDWARE CHECK: -ffast-math is ENABLED." << std::endl;
#else
    std::cout << "HARDWARE CHECK: -ffast-math is NOT enabled." << std::endl;
#endif
    
    std::string batchPrefix = "Test";
    std::string runNumber = "000";
    
    if (argc >= 3) {
        batchPrefix = argv[1];
        runNumber = argv[2];
    }
    
    std::string trajPath = "./" + batchPrefix + "-Trajectory-" + runNumber + ".bin";
    std::string histPath = "./" + batchPrefix + "-Histogram-" + runNumber + ".csv";
    
    std::cout << "Saving trajectory to: " << trajPath << std::endl;
    
    std::string run_id = "000";
    std::string data_dir = "/Users/Dan/Simulations/Computer program 3D/SED_Simulation_Laboratory/data/";
    std::string traj_filename = data_dir + "Trajectory-" + run_id + ".bin";
    std::string hist_filename = data_dir + "Histogram-" + run_id + ".csv";
    
    // ---------------------------------------------------------
    // THE BRIDGE: Reading config.json into the physics engine
    // ---------------------------------------------------------
    std::ifstream configFile("config.json");
    if (!configFile.is_open()) {
        std::cerr << "CRITICAL ERROR: Cannot find config.json!" << std::endl;
        return 1;
    }
    json j_config = json::parse(configFile, nullptr, true, true);
    
    int num_bins_config = j_config["vacuum_parameters"].value("num_frequency_bins", 10000000);

    std::cout << "--- SED Classical Hydrogen Simulation ---" << std::endl;
    std::cout << "Initializing Massive Macroscopic Zero-Point Field Grid (" << num_bins_config << " bins)..." << std::endl;
    
    initializeUniversalGrid(1e14, 1e19, num_bins_config);
    std::cout << "Grid complete!" << std::endl;
    
    SimConfig myConfig;
    
    // Structural toggles
    myConfig.enable_magnetic_lorentz = j_config["simulation_parameters"]["toggles"].value("enable_magnetic_lorentz", true);
    myConfig.enable_coulomb_rr = j_config["simulation_parameters"]["toggles"].value("enable_coulomb_rr", true);
    myConfig.enable_zp = j_config["simulation_parameters"]["toggles"].value("enable_zp", true);
    myConfig.enable_zpp_rr = j_config["simulation_parameters"]["toggles"].value("enable_zpp_rr", false);
    
    myConfig.rng_seed = j_config["simulation_parameters"]["rng_seed"];
    int runOffset = std::stoi(runNumber);
    myConfig.rng_seed = myConfig.rng_seed + runOffset;
    gen.seed(myConfig.rng_seed);
    
    // ---------------------------------------------------------
    // Populating Global Variables from JSON
    // ---------------------------------------------------------
    current_a = j_config["initial_conditions"].value("a_start", 5.291772109e-9);
    current_e = 0.0;
    current_theta = 0.0;
    
    current_w_c = std::sqrt((CHARGE * CHARGE) / (MASS * std::pow(current_a, 3)));
    
    window_spread = j_config["vacuum_parameters"].value("window_width_percent", 0.05);
    taper_width = j_config["vacuum_parameters"].value("taper_width_percent", 0.20);
    
    a_inner = j_config["vacuum_parameters"].value("domain_min_angstroms", 0.05) * 1e-8;
    a_outer = j_config["vacuum_parameters"].value("domain_max_angstroms", 25.0) * 1e-8;
    
    // Integration Parameters
    double t_max = j_config["integration_parameters"]["t_max"];
    double dt = j_config["integration_parameters"]["dt_initial"];
    max_stride = j_config["integration_parameters"].value("tracker_stride", 100);
    myConfig.dt_max = j_config["integration_parameters"].value("dt_max", 5.0e-19);
    myConfig.dt_min = j_config["integration_parameters"].value("dt_min", 1.0e-22);
    myConfig.tolerance = j_config["integration_parameters"].value("tolerance", 1.0e-15);
    
    if (j_config.contains("applied_fields")) {
        for (const auto& field : j_config["applied_fields"]) {
            if (field.value("enabled", false)) {
                AppliedWave wave;
                wave.type = field.value("type", "circular");
                wave.amplitude = field.value("amplitude", 0.0);
                wave.frequency = field.value("frequency", 0.0);
                wave.phase = field.value("phase", 0.0);
                wave.rotation_direction = field.value("rotation_direction", 1);
                myConfig.applied_waves.push_back(wave);
            }
        }
    }
    
    std::cout << "--- DIAGNOSTIC ---" << std::endl;
    std::cout << "Parsed PRNG Seed: " << myConfig.rng_seed << std::endl;
    std::cout << "Starting Radius a: " << (current_a * 1e8) << " A" << std::endl;
    std::cout << "Integration dt:   " << dt << " s" << std::endl;
    std::cout << "------------------" << std::endl;
    
    // Initial State Setup
    State electron;
    electron.x = current_a;
    electron.y = 0.0;
    electron.z = 1.0e-10; // 1 fm tap out of plane
    electron.vx = 0.0;
    electron.vy = std::sqrt((CHARGE * CHARGE) / (MASS * current_a));
    electron.vz = 0.0;
    
    double t = 0.0;
    int acceptedSteps = 0;
    int printInterval = j_config["output_configuration"]["printInterval"];
    
    std::ofstream trajFile(trajPath, std::ios::binary);
    
    if (!trajFile.is_open()) {
        std::cerr << "Error: Could not open binary output file at " << trajPath << "\n";
        return 1;
    }
    
    double snapshot_interval = 25.0e-12;
    double next_snapshot = snapshot_interval;
    
    std::cout << "\nStarting 3D Hydrogen Integration..." << std::endl;
    
    const int num_bins = 2733;
    std::vector<long long> r_histogram(num_bins, 0);
    double r_hist_min = 0.02;
    double r_hist_max = 3.00;
    double dr_bin = (r_hist_max - r_hist_min) / num_bins;
    
    std::string exit_status = "Success! Reached target time.";
    
    while (t < t_max) {
        if (performAdaptiveStepRK4(electron, t, dt, myConfig)) {
            double current_angle = std::atan2(electron.y, electron.x);
            bool orbit_completed = updateRingBuffer(electron.x, electron.y, electron.z, t, current_angle);
            
            if (orbit_completed) {
                double nx = 0, ny = 0, nz = 0;
                double mean_x = 0, mean_y = 0, mean_z = 0;
                
                calculateOrbitalPlaneNormal(orbit_history, FIT_POINTS, nx, ny, nz, mean_x, mean_y, mean_z);
                
                current_theta = std::acos(nz);
                
                double ux, uy, uz, vx, vy, vz;
                if (std::abs(nz) > 0.9) { ux = 0.0; uy = nz; uz = -ny; }
                else { ux = ny; uy = -nx; uz = 0.0; }
                
                double u_mag = std::sqrt(ux*ux + uy*uy + uz*uz);
                ux /= u_mag; uy /= u_mag; uz /= u_mag;
                
                vx = ny * uz - nz * uy;
                vy = nz * ux - nx * uz;
                vz = nx * uy - ny * ux;
                
                std::vector<double> local_u(FIT_POINTS);
                std::vector<double> local_v(FIT_POINTS);
                for (int i = 0; i < FIT_POINTS; ++i) {
                    double dx = orbit_history[i].x - mean_x;
                    double dy = orbit_history[i].y - mean_y;
                    double dz = orbit_history[i].z - mean_z;
                    local_u[i] = (dx * ux) + (dy * uy) + (dz * uz);
                    local_v[i] = (dx * vx) + (dy * vy) + (dz * vz);
                }
                
                double sum_1_r = 0, sum_cos = 0, sum_sin = 0;
                double sum_cos2 = 0, sum_sin2 = 0, sum_sincos = 0;
                double sum_r_cos = 0, sum_r_sin = 0;
                
                for (int i = 0; i < FIT_POINTS; ++i) {
                    double u = local_u[i];
                    double v = local_v[i];
                    double r_loc = std::sqrt(u*u + v*v);
                    double theta_loc = std::atan2(v, u);
                    if (r_loc == 0.0) r_loc = 1e-30;
                    
                    double inv_r = 1.0 / r_loc;
                    double c_th = std::cos(theta_loc);
                    double s_th = std::sin(theta_loc);
                    
                    sum_1_r += inv_r; sum_cos += c_th; sum_sin += s_th;
                    sum_cos2 += (c_th * c_th); sum_sin2 += (s_th * s_th); sum_sincos += (s_th * c_th);
                    sum_r_cos += (inv_r * c_th); sum_r_sin += (inv_r * s_th);
                }
                
                double M11 = 1.0, M12 = sum_cos/FIT_POINTS, M13 = sum_sin/FIT_POINTS;
                double M21 = M12, M22 = sum_cos2/FIT_POINTS, M23 = sum_sincos/FIT_POINTS;
                double M31 = M13, M32 = M23, M33 = sum_sin2/FIT_POINTS;
                double V1 = sum_1_r/FIT_POINTS, V2 = sum_r_cos/FIT_POINTS, V3 = sum_r_sin/FIT_POINTS;
                
                double det = M11*(M22*M33 - M23*M32) - M12*(M21*M33 - M23*M31) + M13*(M21*M32 - M22*M31);
                
                if (std::abs(det) > 1e-30) {
                    double inv11 = (M22*M33 - M23*M32) / det;
                    double inv12 = (M13*M32 - M12*M33) / det;
                    double inv13 = (M12*M23 - M13*M22) / det;
                    double inv21 = (M23*M31 - M21*M33) / det;
                    double inv22 = (M11*M33 - M13*M31) / det;
                    double inv23 = (M13*M21 - M11*M23) / det;
                    double inv31 = (M21*M32 - M22*M31) / det;
                    double inv32 = (M12*M31 - M11*M32) / det;
                    double inv33 = (M11*M22 - M12*M21) / det;
                    
                    double P1 = inv11*V1 + inv12*V2 + inv13*V3;
                    double P2 = inv21*V1 + inv22*V2 + inv23*V3;
                    double P3 = inv31*V1 + inv32*V2 + inv33*V3;
                    
                    double denom = (P2*P2 + P3*P3);
                    if (denom > 0 && P1 > 0) {
                        double p = std::sqrt(1.0 / denom);
                        double eps = 1.0 / (P1 * p);
                        if (eps < 0.0) eps = 0.0;
                        if (eps >= 1.0) eps = 0.999999;
                        
                        current_a = 1.0 / (P1 * (1.0 - eps * eps));
                        current_e = eps;
                        
                        double a_cubed = std::abs(current_a * current_a * current_a);
                        current_w_c = std::sqrt((CHARGE * CHARGE) / (MASS * a_cubed));
                    }
                }
            }
            
            if (acceptedSteps % printInterval == 0) {
                double r = std::sqrt(electron.x*electron.x + electron.y*electron.y + electron.z*electron.z);
                double v2 = electron.vx*electron.vx + electron.vy*electron.vy + electron.vz*electron.vz;
                double current_energy = 0.5 * MASS * v2 - (CHARGE * CHARGE) / r;
                
                double Lx = MASS * (electron.y * electron.vz - electron.z * electron.vy);
                double Ly = MASS * (electron.z * electron.vx - electron.x * electron.vz);
                double Lz = MASS * (electron.x * electron.vy - electron.y * electron.vx);
                double L_mag = std::sqrt(Lx*Lx + Ly*Ly + Lz*Lz);
                
                double output_data[8] = {t, r, current_a, current_e, current_theta, current_energy, L_mag, Lz};
                trajFile.write(reinterpret_cast<const char*>(&output_data), sizeof(output_data));
            }
            
            acceptedSteps++;
            
            if (acceptedSteps > 50000) {
                double r_angstroms = std::sqrt(electron.x*electron.x + electron.y*electron.y + electron.z*electron.z) * 1e8;
                if (r_angstroms >= r_hist_min && r_angstroms < r_hist_max) {
                    int bin_index = static_cast<int>((r_angstroms - r_hist_min) / dr_bin);
                    r_histogram[bin_index]++;
                }
            }
            
            if (acceptedSteps % 10000 == 0) {
                std::cout << "Time: " << std::scientific << t << " s | "
                << "a = " << std::fixed << std::setprecision(4) << (current_a * 1e8) << " A | "
                << "ecc = " << std::fixed << std::setprecision(5) << current_e << " | "
                << "Theta: " << std::fixed << std::setprecision(15) << current_theta << std::endl;
            }
            
            double v2_check = electron.vx*electron.vx + electron.vy*electron.vy + electron.vz*electron.vz;
            double r_check = std::sqrt(electron.x*electron.x + electron.y*electron.y + electron.z*electron.z);
            double current_energy = 0.5 * MASS * v2_check - (CHARGE * CHARGE) / r_check;
            
            if (r_check < 5.0e-12) {
                exit_status = "DECAYED: Orbit collapsed into nucleus";
                break;
            }
            else if (current_energy >= 0.0 && r_check > 5.0e-8) {
                exit_status = "IONIZED: Electron escaped";
                break;
            }
            else if (r_check > 25.0e-8) {
                exit_status = "LOST: Electron drifted beyond boundary";
                break;
            }
        } else {
            exit_status = "INTEGRATOR CRASH: Adaptive stepper failed";
            break;
        }
        
        // --- PERIODIC HISTOGRAM SNAPSHOT ---
        if (t >= next_snapshot) {
            int current_ps = static_cast<int>(t * 1e12);
            std::string snap_filename = batchPrefix + "-Hist-" + std::to_string(current_ps) + "ps-" + runNumber + ".csv";
            std::string snap_path = "./" + snap_filename;
            
            std::ofstream snap_outfile(snap_path);
            snap_outfile << "r_angstroms,counts\n";
            for (int i = 0; i < num_bins; ++i) {
                double r_val = r_hist_min + (i * dr_bin) + (dr_bin / 2.0);
                snap_outfile << r_val << "," << r_histogram[i] << "\n";
            }
            snap_outfile.close();
            
            next_snapshot += snapshot_interval;
        }
    }
    
    std::cout << "\nExporting P(r) data to " << hist_filename << "..." << std::endl;
    std::ofstream csv_outfile(histPath);
    csv_outfile << "r_angstroms,counts\n";
    for (int i = 0; i < num_bins; ++i) {
        double r_val = r_hist_min + (i * dr_bin) + (dr_bin / 2.0);
        csv_outfile << r_val << "," << r_histogram[i] << "\n";
    }
    csv_outfile.close();
    
    trajFile.close();
    
    std::cout << "\n" << exit_status << " (Steps: " << acceptedSteps << ")\n";
    return 0;
}
