#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "symmetry/su2_handler.h"

using namespace tapp_con;

TEST_CASE("SU2Handler — compatibility via triangle inequality") {
    SUBCASE("1/2 x 1/2 -> {0, 1}") {
        // spin-1/2 x 2 = 1, spin-0 x 2 = 0, spin-1 x 2 = 2
        CHECK(SU2Handler::compatible(1, 1, 0)); // 1/2 x 1/2 -> 0
        CHECK(SU2Handler::compatible(1, 1, 2)); // 1/2 x 1/2 -> 1
        CHECK_FALSE(SU2Handler::compatible(1, 1, 1)); // 1/2 is not in product
        CHECK_FALSE(SU2Handler::compatible(1, 1, 4)); // too large
    }

    SUBCASE("1 x 1 -> {0, 1, 2}") {
        CHECK(SU2Handler::compatible(2, 2, 0));
        CHECK(SU2Handler::compatible(2, 2, 2));
        CHECK(SU2Handler::compatible(2, 2, 4));
        CHECK_FALSE(SU2Handler::compatible(2, 2, 6));
    }

    SUBCASE("1 x 1/2 -> {1/2, 3/2}") {
        CHECK(SU2Handler::compatible(2, 1, 1));
        CHECK(SU2Handler::compatible(2, 1, 3));
        CHECK_FALSE(SU2Handler::compatible(2, 1, 5));
    }

    SUBCASE("1/2 x 0 -> {1/2}") {
        CHECK(SU2Handler::compatible(1, 0, 1));
        CHECK_FALSE(SU2Handler::compatible(1, 0, 3));
    }
}

TEST_CASE("SU2Handler — allowed_fusions") {
    SUBCASE("1/2 x 1/2") {
        auto fusions = SU2Handler::allowed_fusions(1, 1);
        REQUIRE(fusions.size() == 2);
        CHECK(fusions[0] == 0); // spin-0
        CHECK(fusions[1] == 2); // spin-1
    }

    SUBCASE("1 x 1") {
        auto fusions = SU2Handler::allowed_fusions(2, 2);
        REQUIRE(fusions.size() == 3);
        CHECK(fusions[0] == 0);
        CHECK(fusions[1] == 2);
        CHECK(fusions[2] == 4);
    }

    SUBCASE("1 x 1/2") {
        auto fusions = SU2Handler::allowed_fusions(2, 1);
        REQUIRE(fusions.size() == 2);
        CHECK(fusions[0] == 1);
        CHECK(fusions[1] == 3);
    }
}

TEST_CASE("SU2Handler — placeholder CG coefficients") {
    // Phase 1: all valid fusions return 1.0
    double cg = SU2Handler::clebsch_gordan(1, 1, 1, -1, 0, 0);
    CHECK(cg == 1.0);

    // Any valid fusion is fine
    cg = SU2Handler::clebsch_gordan(2, 2, 1, 1, 3, 3);
    CHECK(cg == 1.0);
}
