#include "constraint_parser.h"
#include <cctype>
#include <sstream>
#include <cstdlib>
#include <algorithm>

namespace tapp_con {

// ── Tokenizer ──────────────────────────────────────────

enum class TokenType {
    IDENT, NUMBER, PLUS, MINUS, STAR, EQUALS, END
};

struct Token {
    TokenType type;
    std::string text;
    int64_t value; // for NUMBER tokens
};

static std::vector<Token> tokenize(const std::string& expr, std::string* error) {
    std::vector<Token> tokens;
    size_t i = 0;
    while (i < expr.size()) {
        char c = expr[i];
        if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
            continue;
        }
        if (c == '+') { tokens.push_back({TokenType::PLUS, "+", 0}); ++i; continue; }
        if (c == '-') { tokens.push_back({TokenType::MINUS, "-", 0}); ++i; continue; }
        if (c == '*') { tokens.push_back({TokenType::STAR, "*", 0}); ++i; continue; }
        if (c == '=') { tokens.push_back({TokenType::EQUALS, "=", 0}); ++i; continue; }
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '-') {
            // Also handle negative numbers
            bool neg = false;
            if (c == '-') {
                neg = true;
                ++i;
                if (i >= expr.size() || !std::isdigit(static_cast<unsigned char>(expr[i]))) {
                    // This '-' is an operator, push it and continue
                    tokens.push_back({TokenType::MINUS, "-", 0});
                    continue;
                }
            }
            int64_t val = 0;
            while (i < expr.size() && std::isdigit(static_cast<unsigned char>(expr[i]))) {
                val = val * 10 + (expr[i] - '0');
                ++i;
            }
            if (neg) val = -val;
            tokens.push_back({TokenType::NUMBER, std::to_string(val), val});
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t start = i;
            while (i < expr.size() && (std::isalnum(static_cast<unsigned char>(expr[i])) || expr[i] == '_')) {
                ++i;
            }
            tokens.push_back({TokenType::IDENT, expr.substr(start, i - start), 0});
            continue;
        }
        if (error) *error = "Unexpected character '" + std::string(1, c) + "'";
        return {};
    }
    tokens.push_back({TokenType::END, "", 0});
    return tokens;
}

// ── Parser ─────────────────────────────────────────────

/// Parse: expr = term { (+|-) term }
/// Term: [coefficient *] ident
static bool parse_expr(const std::vector<Token>& tokens, size_t& pos,
                       LinearConstraint& lc, std::string* error) {
    int32_t sign = +1;

    while (pos < tokens.size()) {
        if (tokens[pos].type == TokenType::END || tokens[pos].type == TokenType::EQUALS)
            break;

        // Optional sign
        if (tokens[pos].type == TokenType::PLUS) {
            sign = +1;
            ++pos;
        } else if (tokens[pos].type == TokenType::MINUS) {
            sign = -1;
            ++pos;
        }

        // Parse term: [NUMBER *] IDENT
        int32_t coefficient = sign;

        if (tokens[pos].type == TokenType::NUMBER) {
            coefficient = sign * static_cast<int32_t>(tokens[pos].value);
            ++pos;
            if (pos >= tokens.size() || tokens[pos].type != TokenType::STAR) {
                if (error) *error = "Expected '*' after coefficient";
                return false;
            }
            ++pos; // skip '*'
        }

        // Must be ident
        if (pos >= tokens.size() || tokens[pos].type != TokenType::IDENT) {
            if (error) *error = "Expected identifier (q<leg>)";
            return false;
        }

        // Parse "q" + digit(s)
        const std::string& id = tokens[pos].text;
        if (id.size() < 2 || id[0] != 'q') {
            if (error) *error = "Expected identifier of form 'q<leg_index>'";
            return false;
        }
        int leg_idx = 0;
        for (size_t ci = 1; ci < id.size(); ++ci) {
            if (!std::isdigit(static_cast<unsigned char>(id[ci]))) {
                if (error) *error = "Invalid leg index in '" + id + "'";
                return false;
            }
            leg_idx = leg_idx * 10 + (id[ci] - '0');
        }
        ++pos;

        lc.terms.push_back({leg_idx, coefficient});
        sign = +1; // reset after term
    }

    return true;
}

LinearConstraint ConstraintParser::parse_u1(const std::string& expr,
                                             std::string* error) {
    LinearConstraint lc;

    auto tokens = tokenize(expr, error);
    if (tokens.empty()) return lc;

    size_t pos = 0;

    // Parse left-hand side expression
    if (!parse_expr(tokens, pos, lc, error)) return lc;

    // Expect '='
    if (pos >= tokens.size() || tokens[pos].type != TokenType::EQUALS) {
        if (error) *error = "Expected '='";
        return lc;
    }
    ++pos;

    // Parse target value
    if (pos >= tokens.size() || tokens[pos].type != TokenType::NUMBER) {
        if (error) *error = "Expected integer target value";
        return lc;
    }
    lc.target = static_cast<int32_t>(tokens[pos].value);
    ++pos;

    // Check for trailing tokens
    if (pos < tokens.size() - 1 || (pos < tokens.size() && tokens[pos].type != TokenType::END)) {
        if (error) *error = "Unexpected trailing characters";
        return lc;
    }

    return lc;
}

ParseResult ConstraintParser::parse_system(
    uint64_t n_legs,
    const std::vector<std::string>& u1_exprs,
    const std::vector<int>& sector_sizes,
    SymmetryType sym_type,
    int32_t charge_min,
    int32_t charge_max,
    const std::vector<std::vector<int32_t>>& spin_sectors)
{
    ParseResult result;
    result.success = false;
    result.system.sym_type = sym_type;

    // Parse each U(1) constraint string
    for (const auto& expr : u1_exprs) {
        std::string err;
        auto lc = parse_u1(expr, &err);
        if (!err.empty()) {
            result.error_message = "Parse error in '" + expr + "': " + err;
            return result;
        }
        result.system.u1_constraints.push_back(std::move(lc));
    }

    // Generate leg sectors
    bool has_u1 = (static_cast<uint8_t>(sym_type) & static_cast<uint8_t>(SymmetryType::U1)) != 0;
    bool has_su2 = (static_cast<uint8_t>(sym_type) & static_cast<uint8_t>(SymmetryType::SU2)) != 0;

    for (uint64_t leg = 0; leg < n_legs; ++leg) {
        LegSector ls;
        ls.leg_index = static_cast<int>(leg);

        if (has_u1) {
            int num = sector_sizes[leg];
            if (charge_max - charge_min + 1 < num) {
                // Need more sectors than range allows — distribute evenly
                int step = std::max(1, (charge_max - charge_min) / std::max(1, num - 1));
                for (int i = 0; i < num; ++i) {
                    ls.allowed_charges.push_back(charge_min + i * step);
                }
            } else {
                // Evenly spaced over range
                for (int i = 0; i < num; ++i) {
                    int32_t q = charge_min + (i * (charge_max - charge_min)) / std::max(1, num - 1);
                    ls.allowed_charges.push_back(q);
                }
            }
        } else {
            ls.allowed_charges.push_back(0);
        }

        if (has_su2 && leg < spin_sectors.size() && !spin_sectors[leg].empty()) {
            ls.allowed_spins_t2 = spin_sectors[leg];
        } else if (has_su2) {
            // Default: spin-0 only
            ls.allowed_spins_t2.push_back(0);
        } else {
            ls.allowed_spins_t2.push_back(0);
        }

        // Verify alignment
        if (!ls.allowed_spins_t2.empty() && !ls.allowed_charges.empty() &&
            ls.allowed_spins_t2.size() != ls.allowed_charges.size()) {
            // If misaligned, repeat the single spin value for each charge
            int32_t s = ls.allowed_spins_t2[0];
            ls.allowed_spins_t2.assign(ls.allowed_charges.size(), s);
        }

        result.system.leg_sectors.push_back(std::move(ls));
    }

    result.success = true;
    return result;
}

} // namespace tapp_con
