// Standalone test: verify jacobi_eigen at d=768 vs numpy eigh.
// Build: see scripts below. Reads a covariance matrix from stdin (raw double),
// writes eigenvalues then eigenvectors to stdout (raw double).
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>

static void jacobi_eigen(std::vector<double>& a, uint32_t dim,
                         std::vector<double>& eigvals,
                         std::vector<double>* eigvecs = nullptr) {
    if (eigvecs) {
        eigvecs->assign(size_t(dim) * dim, 0.0);
        for (uint32_t i = 0; i < dim; i++) (*eigvecs)[i * dim + i] = 1.0;
    }
    for (uint32_t sweep = 0; sweep < 50; sweep++) {
        double off = 0.0;
        for (uint32_t p = 0; p < dim; p++)
            for (uint32_t q = p + 1; q < dim; q++)
                off += std::fabs(a[p * dim + q]);
        if (off < 1e-12) break;
        for (uint32_t p = 0; p < dim; p++) {
            for (uint32_t q = p + 1; q < dim; q++) {
                const double apq = a[p * dim + q];
                if (std::fabs(apq) < 1e-15) continue;
                const double app = a[p * dim + p];
                const double aqq = a[q * dim + q];
                double theta = (aqq - app) / (2.0 * apq);
                double t;
                if (std::fabs(theta) > 1e15) {
                    t = 1.0 / (2.0 * theta);
                } else {
                    t = (theta >= 0.0 ? 1.0 : -1.0) /
                        (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                }
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                a[p * dim + p] = app - t * apq;
                a[q * dim + q] = aqq + t * apq;
                a[p * dim + q] = 0.0;
                a[q * dim + p] = 0.0;
                for (uint32_t i = 0; i < dim; i++) {
                    if (i == p || i == q) continue;
                    const double aip = a[i * dim + p];
                    const double aiq = a[i * dim + q];
                    a[i * dim + p] = c * aip - s * aiq;
                    a[p * dim + i] = a[i * dim + p];
                    a[i * dim + q] = s * aip + c * aiq;
                    a[q * dim + i] = a[i * dim + q];
                }
                if (eigvecs) {
                    for (uint32_t i = 0; i < dim; i++) {
                        const double vip = (*eigvecs)[i * dim + p];
                        const double viq = (*eigvecs)[i * dim + q];
                        (*eigvecs)[i * dim + p] = c * vip - s * viq;
                        (*eigvecs)[i * dim + q] = s * vip + c * viq;
                    }
                }
            }
        }
    }
    eigvals.resize(dim);
    for (uint32_t i = 0; i < dim; i++) eigvals[i] = a[i * dim + i];
}

int main(int argc, char** argv) {
    // Mode: read dim then covariance from stdin (raw double), time eigendecomp,
    // write eigvals + eigvecs to stdout. argv[1] optional seed for synthetic.
    uint32_t dim;
    if (fread(&dim, sizeof(uint32_t), 1, stdin) != 1) return 1;
    std::vector<double> cov(size_t(dim) * dim);
    if (fread(cov.data(), sizeof(double), size_t(dim) * dim, stdin) !=
        size_t(dim) * dim) {
        return 1;
    }
    std::vector<double> eigvals, eigvecs;
    jacobi_eigen(cov, dim, eigvals, &eigvecs);
    // Write eigvals then eigvecs.
    fwrite(eigvals.data(), sizeof(double), dim, stdout);
    fwrite(eigvecs.data(), sizeof(double), size_t(dim) * dim, stdout);
    return 0;
}
