#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "symmetry/u1_handler.h"

using namespace tapp_con;

TEST_CASE("U1Handler — compatibility") {
    SUBCASE("conservation: qA + qB = qC") {
        CHECK(U1Handler::compatible(0, 0, 0));
        CHECK(U1Handler::compatible(1, 2, 3));
        CHECK(U1Handler::compatible(-1, 1, 0));
        CHECK(U1Handler::compatible(5, -3, 2));
    }

    SUBCASE("non-conserving combinations") {
        CHECK_FALSE(U1Handler::compatible(0, 0, 1));
        CHECK_FALSE(U1Handler::compatible(1, 1, 3));
        CHECK_FALSE(U1Handler::compatible(-1, -1, 0));
        CHECK_FALSE(U1Handler::compatible(2, 3, 4));
    }
}

TEST_CASE("U1Handler — fuse") {
    CHECK(U1Handler::fuse(0, 0) == 0);
    CHECK(U1Handler::fuse(1, 2) == 3);
    CHECK(U1Handler::fuse(-1, 1) == 0);
    CHECK(U1Handler::fuse(100, -100) == 0);
}
