#pragma once

#include "nn/Identity.hpp"

class Dropout : public Identity {
public:
    Dropout(float) {}
};
