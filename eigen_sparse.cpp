#include <Eigen/Sparse>
#include <vector>
#include <CImg.h>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

void insertCoefficient(int id, int i, int j, double w, std::vector<T>& coeffs,
                       Eigen::VectorXd& b, const Eigen::VectorXd& boundary)
{
  int n = boundary.size();
  int id1 = i+j*n;
        if(i==-1 || i==n) b(id) -= w * boundary(j); // constrained coeffcieint
  else  if(j==-1 || j==n) b(id) -= w * boundary(i); // constrained coeffcieint
  else  coeffs.push_back(T(id,id1,w));              // unknown coefficient
}

void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n)
{
  b.setZero();
  Eigen::ArrayXd boundary = Eigen::ArrayXd::LinSpaced(n, 0,M_PI).sin().pow(2);
  for(int j=0; j<n; ++j)
  {
    for(int i=0; i<n; ++i)
    {
      int id = i+j*n;
      insertCoefficient(id, i-1,j, -1, coefficients, b, boundary);
      insertCoefficient(id, i+1,j, -1, coefficients, b, boundary);
      insertCoefficient(id, i,j-1, -1, coefficients, b, boundary);
      insertCoefficient(id, i,j+1, -1, coefficients, b, boundary);
      insertCoefficient(id, i,j,    4, coefficients, b, boundary);
    }
  }
}

void saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename)
{
  Eigen::Array<unsigned char,Eigen::Dynamic,Eigen::Dynamic> bits = (x*255).cast<unsigned char>();
  cimg_library::CImg<unsigned char> img(bits.data(), n,n);  
  img.display();
}

int main(int argc, char** argv)
{
  int n = 1000;  // size of the image
  int m = n*n;  // number of unknows (=number of pixels)
  // Assembly:
  std::vector<T> coefficients;            // list of non-zeros coefficients
  Eigen::VectorXd b(m);                   // the right hand side-vector resulting from the constraints
  buildProblem(coefficients, b, n);
  SpMat A(m,m);
  A.setFromTriplets(coefficients.begin(), coefficients.end());
  // Solving:
  Eigen::SimplicialCholesky<SpMat> chol(A);  // performs a Cholesky factorization of A
  Eigen::VectorXd x = chol.solve(b);         // use the factorization to solve for the given right hand side
  // Export the result to a file:
  saveAsBitmap(x, n, argv[1]);
  return 0;
}

