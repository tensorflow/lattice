/* Copyright 2017 The TensorFlow Lattice Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow_lattice/cc/kernels/monotonic_lattice_projections.h"

#include <vector>

#include "tensorflow_lattice/cc/lib/lattice_structure.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace lattice {
namespace {
TEST(MonotoneLatticeProjectorErrorTest, ProjectionWithNullptr) {
  LatticeStructure lattice_structure(/*lattice_sizes=*/{2, 2});
  MonotoneLatticeProjector<float> projector(lattice_structure,
                                            /*monotone_dimensions=*/{});
  const Status s =
      projector.Project(/*lattice_param_vec=*/{0, 1, 2, 3}, nullptr);
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
}

TEST(MonotoneLatticeProjectorErrorTest, ProjectionWithWrongInputDimension) {
  LatticeStructure lattice_structure(/*lattice_sizes=*/{2, 2});
  MonotoneLatticeProjector<float> projector(lattice_structure,
                                            /*monotone_dimensions=*/{});
  std::vector<float> output(4, 0.0);
  const Status s = projector.Project(/*lattice_param_vec=*/{0, 1, 2}, &output);
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
}

TEST(MonotoneLatticeProjectorErrorTest, ProjectionWithWrongOutputDimension) {
  LatticeStructure lattice_structure(/*lattice_sizes=*/{2, 2});
  MonotoneLatticeProjector<float> projector(lattice_structure,
                                            /*monotone_dimensions=*/{});
  std::vector<float> output(3, 0.0);
  const Status s =
      projector.Project(/*lattice_param_vec=*/{0, 1, 2, 3}, &output);
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
}

// The fixture for testing MonotoneLatticeProjector.
class MonotoneLatticeProjectorTest : public ::testing::Test {
 protected:
  void CheckProjection(
      const std::vector<int>& lattice_sizes,
      const std::vector<int>& monotone_dimensions,
      const std::vector<float>& lattice_param_vec,
      const std::vector<float>& expected_projected_lattice_param_vec) {
    LatticeStructure lattice_structure(lattice_sizes);
    MonotoneLatticeProjector<float> projector(lattice_structure,
                                              monotone_dimensions, kEpsilon);
    std::vector<float> projected_lattice_param_vec(lattice_param_vec.size());
    TF_ASSERT_OK(
        projector.Project(lattice_param_vec, &projected_lattice_param_vec));
    LOG(INFO) << "lattice param: " << str_util::Join(lattice_param_vec, ",");
    LOG(INFO) << "Expected projected lattice param: "
              << str_util::Join(expected_projected_lattice_param_vec, ",");
    LOG(INFO) << "Projected lattice param: "
              << str_util::Join(projected_lattice_param_vec, ",");

    ASSERT_EQ(projected_lattice_param_vec.size(),
              expected_projected_lattice_param_vec.size());
    for (int ii = 0; ii < expected_projected_lattice_param_vec.size(); ++ii) {
      EXPECT_NEAR(expected_projected_lattice_param_vec[ii],
                  projected_lattice_param_vec[ii], kEpsilon);
    }
  }

 private:
  const float kEpsilon = 1e-5;
};

TEST_F(MonotoneLatticeProjectorTest, ProjectToNothing) {
  CheckProjection(
      /*lattice_sizes=*/{2, 2}, /*monotone_dimensions=*/{},
      /*lattice_param_vec=*/{3.0, 0.0, 2.0, 5.0},
      /*expected_projected_lattice_param_vec=*/{3.0, 0.0, 2.0, 5.0});
}

TEST_F(MonotoneLatticeProjectorTest, ProjectTo0thDimension) {
  CheckProjection(
      /*lattice_sizes=*/{2, 2}, /*monotone_dimensions=*/{0},
      /*lattice_param_vec=*/{3.0, 0.0, 2.0, 5.0},
      /*expected_projected_lattice_param_vec=*/{1.5, 1.5, 2.0, 5.0});
}

TEST_F(MonotoneLatticeProjectorTest, ProjectTo1stDimension) {
  CheckProjection(
      /*lattice_sizes=*/{2, 2}, /*monotone_dimensions=*/{1},
      /*lattice_param_vec=*/{3.0, 0.0, 2.0, 5.0},
      /*expected_projected_lattice_param_vec=*/{2.5, 0.0, 2.5, 5.0});
}

TEST_F(MonotoneLatticeProjectorTest, ProjectToAllDimensions) {
  CheckProjection(
      /*lattice_sizes=*/{2, 2}, /*monotone_dimensions=*/{0, 1},
      /*lattice_param_vec=*/{3.0, 0.0, 2.0, 5.0},
      /*expected_projected_lattice_param_vec=*/{1.5, 1.5, 2.0, 5.0});
}

TEST_F(MonotoneLatticeProjectorTest, ProjectThreeByTwoLatticeToAllDimensions) {
  CheckProjection(
      /*lattice_sizes=*/{3, 2}, /*monotone_dimensions=*/{0, 1},
      /*lattice_param_vec=*/{3.0, 1.0, 0.0, 0.0, 2.0, 5.0},
      /*expected_projected_lattice_param_vec=*/{1.0, 1.0, 1.0, 1.0, 2.0, 5.0});
}

TEST_F(MonotoneLatticeProjectorTest,
       ProjectTwoByTwoByTwoLatticeToAllDimensions) {
  CheckProjection(
      /*lattice_sizes=*/{2, 2, 2},
      /*monotone_dimensions=*/{0, 1, 2},
      /*lattice_param_vec=*/{0.44, 0.3, 0.12, 3.33, 3.0, 0.0, 2.0, 5.0},
      /*expected_projected_lattice_param_vec=*/{0.28, 0.3, 0.28, 3.33, 1.5, 1.5,
                                                2.0, 5.0});
}

}  // namespace
}  // namespace lattice
}  // namespace tensorflow
