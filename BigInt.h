#pragma once

#include <iostream>
#include <cmath>
#include <string>
#include <algorithm>
#include <vector>
#include <thread>
#include <cstdint>
#include <intrin.h>
#include <immintrin.h>
#include "Benchmark.h"
#include <random>

typedef uint64_t var_type;
const uint64_t BASE = 268435456; // 65536, 536870912, 2147483648, 4294967296, 18446744073709551616
const uint8_t EXPONENT = 28;
const int KARATSUBA_LIMIT = 70;
const int MULTI_THREAD_LIMIT = 10000;
const bool MULTI_THREAD = true;
const uint64_t STACK_LIMIT = 4096;

std::random_device RndDevice;
std::default_random_engine RndEngine(RndDevice());
std::uniform_int_distribution<uint64_t> Dist(0, BASE - 1);

static void* MarkAlloct(void* p, uint8_t mark) {
	static_cast<uint8_t*>(p)[0] = mark;
	return static_cast<uint8_t*>(p) + 32;
}

static void _free_alloct(void* p) {
	uint8_t* bytes = static_cast<uint8_t*>(p) - 32;

	if (bytes[0] == 1) {
		_aligned_free(bytes);
	}
}
#define _alloct(size)															\
		__pragma(warning(suppress: 6255))										\
		(size <= STACK_LIMIT													\
		? MarkAlloct((void*)(((uint64_t)_alloca(size + 63) + 31) & ~31), 0)		\
		: MarkAlloct(_aligned_malloc(size + 32, 32), 1))		                \

class BigInt {
private:
	void Clean() {
		if (IsZero()) {
			value.resize(1);
			value[0] = 0;
			sign = 1;
		}

		size_t count = 0;
		while ((value.size() - count) > 1 && value[value.size() - (1 + count)] == 0) {
			count++;
		}
		if (!count) { return; }

		value.erase(value.end() - count, value.end());
		
		if (IsZero())
			sign = 1;
	}
public:
	std::vector <var_type> value;
	int8_t sign = 1;

	BigInt() {

	}

	BigInt(int64_t num) {
		if (num == 0) {
			value.resize(1);
			value[0] = 0;
			return;
		}
		else if (num < 0) {
			num *= -1;
			sign = -1;
		}
		if (num < BASE) {
			value.push_back(num);
			return;
		}

		size_t count = size_t(log10(num) / log10(BASE) + 1);
		value.reserve(count);
		while (num > 0) {
			value.push_back(num % BASE);
			num /= BASE;
		}
	}

	BigInt(const std::string& num) {
		const BigInt TEN(10);

		size_t i = 0, count = num.length();
		BigInt n, result = 0;

		if (num[0] == '-') {
			sign = -1;
			i++;
		}

		for (i; i < count; ++i) {
			result *= TEN;
			n = num[i] - '0';
			result += n;
		}
		value = std::move(result.value);
	}

	BigInt(const BigInt& num) {
		value = num.value;
		sign = num.sign;
	}

	BigInt(BigInt&& num) noexcept {
		value = std::move(num.value);
		sign = num.sign;
	}

	BigInt& operator+=(const BigInt& other) {
		if (sign != other.sign) {
			sign = (sign > other.sign || *this < other ? 1 : -1);
			if (*this >= other) {
				return Subtract(other);
			}
			else {
				BigInt res = other;
				res.Subtract(*this);
				value = std::move(res.value);
				return *this;
			}
		}

		return Add(other);
	}

	BigInt& Add(const BigInt& other) {
		const size_t m = other.value.size(), max = (value.size() > m ? value.size() : m) + 1;

		value.resize(max);

		uint8_t add = 0;
		for (size_t i = 0; i < max; ++i) {
			uint64_t c = value[i] + ((m > i) ? other.value[i] : 0);
			value[i] = (c + add) & (BASE - 1);
			add = (c + add) >> EXPONENT;
		}

		Clean();
		return *this;
	}

	friend BigInt operator+(const BigInt& lhs, const BigInt& rhs) {
		BigInt result(lhs);
		result += rhs;
		return result;
	}

	BigInt& operator-=(const BigInt& other) {
		if (sign == other.sign) {
			sign = (*this > other ? 1 : -1);
			sign *= other.sign;

			Difference(other);
		}
		else {
			Add(other);
		}

		return *this;
	}

	BigInt& Difference(const BigInt& other) {
		size_t max = value.size();
		BigInt a, b;
		a.value = value;
		b.value = other.value;

		if (b > a) {
			max = other.value.size();
			std::swap(a, b);
			value.resize(max);
		}

		a.Subtract(b);

		value = std::move(a.value);

		return *this;
	}

	BigInt& Subtract(const BigInt& other) {
		const size_t n = value.size(), m = other.value.size();
		int64_t k = 0, t;
		for (size_t i = 0; i < n; ++i) {
			t = value[i] - (i >= m ? 0 : other.value[i]) + k;
			value[i] = t & (BASE - 1);
			k = t >> EXPONENT;
		}

		Clean();
		return *this;
	}

	friend BigInt operator-(const BigInt& lhs, const BigInt& rhs) {
		BigInt result(lhs);
		result -= rhs;
		return result;
	}

	BigInt& operator*=(const BigInt& other) {
		if (IsZero() || other.IsZero()) {
			sign = 1;
			value.resize(1);
			value[0] = 0;
			return * this;
		}
		sign *= other.sign;

#ifdef __AVX2__
		if ((value.size() > 254 && other.value.size() > 254))
			return KMul(other);
		else
			return *this == other ? Square_I() : BaseMul_I(other);
#else
		if ((value.size() > KARATSUBA_LIMIT && other.value.size() > KARATSUBA_LIMIT))
			return KMul(other);
		else
			return BaseMul(other);
#endif // __AVX2__
	}

	friend BigInt operator*(const BigInt& lhs, const BigInt& rhs) {
		BigInt result(lhs);
		result *= rhs;
		return result;
	}

	BigInt& BaseMul(const BigInt& other) {
		const size_t n = value.size();
		const size_t m = other.value.size();
		uint64_t carry, val;
		size_t i, j;

		std::vector<var_type> copy = std::move(value);
		value.resize(n + m);
		for (i = 0; i < m; ++i) {
			carry = 0;
			for (j = 0; j < n; ++j) {
				val = value[i + j];
				val += carry;
				val += copy[j] * other.value[i];
				value[i + j] = val & (BASE - 1);
				carry = val >> EXPONENT;
			}

			value[i + n] = carry;
		}
		Clean();

		return *this;
	}

	BigInt& BaseMul_I(const BigInt& other) {
		const size_t n = value.size();
		const size_t m = other.value.size();
		size_t i, j;
		__m256i a, b, c;

		uint64_t* result = (uint64_t*)_alloct((n + m) * sizeof(uint64_t));
		uint64_t* copy = (uint64_t*)_alloct(n * sizeof(uint64_t));
		
		memset(&result[0], 0, (n + m) * sizeof(uint64_t));
		memcpy(&copy[0], &value[0], n * sizeof(uint64_t));

		_mm_prefetch((const char*)&result, _MM_HINT_T0);
		_mm_prefetch((const char*)&copy, _MM_HINT_T0);

		for (i = 0; i < m; ++i) {
			b = _mm256_set1_epi32(other.value[i]);
			for (j = 0; j + 3 < n; j += 4) {
				a = _mm256_load_si256(reinterpret_cast<__m256i*>(&copy[j]));
				c = _mm256_mul_epu32(a, b);
				
				a = _mm256_load_si256(reinterpret_cast<__m256i*>(&result[i + j]));
				c = _mm256_add_epi64(a, c);

				_mm256_store_si256(reinterpret_cast<__m256i*>(&result[i + j]), c);
			}
			for (j; j < n; ++j) {
				result[i + j] += value[j] * other.value[i];
			}
		}

		uint64_t carry = 0;
		for (i = 0; i < n + m; ++i) {
			result[i] += carry;
			carry = result[i];
			result[i] &= (BASE - 1);
			carry >>= EXPONENT;
		}

		size_t size = n + m - (result[n + m - 1] == 0 ? 1 : 0);

		value.resize(size);
		memcpy(&value[0], &result[0], size * sizeof(uint64_t));
		_free_alloct(result);
		return *this;
	}

	BigInt& Square() {
		std::vector<var_type> result;
		uint64_t uv, c, v, u;
		size_t t = value.size();
		result.resize(t * 2);
		for (size_t i = 0; i < t; ++i) {
			uv = result[2 * i] + value[i] * value[i];
			v = uv % BASE;
			u = (uv - v) / BASE;
			result[2 * i] = v;
			c = u;
			for (size_t j = i + 1; j < t; ++j) {
				uv = result[i + j] + 2 * value[j] * value[i] + c;
				v = uv % BASE;
				u = (uv - v) / BASE;
				result[i + j] = v;
				c = u;
			}
			result[i + t] = u;
		}

		value = std::move(result);
		Clean();
		return *this;
	}

	BigInt& Square_I() {
		const size_t n = value.size(), n2 = n << 1;
		size_t i, j;
		__m256i a, b, c, r;

		uint64_t* result = (uint64_t*)_alloca(n2 * sizeof(uint64_t) + 31);
		result = (uint64_t*)((((uint64_t)result + 31) & ~31));
		memset(&result[0], 0, n2 * sizeof(uint64_t));

		uint64_t* copy = (uint64_t*)_alloca(n * sizeof(uint64_t) + 31);
		copy = (uint64_t*)((((uint64_t)copy + 31) & ~31));
		memcpy(&copy[0], &value[0], n * sizeof(uint64_t));

		for (i = 0; i < n; i += 1) {
			result[i << 1] += copy[i] * copy[i];
			a = _mm256_set1_epi32(value[i]);

			for (j = i + 1; j + 3 < n; j += 4) {
				b = _mm256_load_si256(reinterpret_cast<__m256i*>(&copy[j]));

				c = _mm256_mul_epu32(a, b);
				c = _mm256_slli_epi64(c, 1);

				r = _mm256_load_si256(reinterpret_cast<__m256i*>(&result[i + j]));
				c = _mm256_add_epi64(r, c);

				_mm256_store_si256(reinterpret_cast<__m256i*>(&result[i + j]), c);
			}
			for (j; j < n; ++j) {
				result[i + j] += (copy[i] * copy[j]) << 1;
			}
		}

		uint64_t carry = 0;
		for (i = 0; i < n2; ++i) {
			result[i] += carry;
			carry = result[i];
			result[i] &= BASE - 1;
			carry >>= EXPONENT;
		}

		size_t size = n2 - (result[n2 - 1] == 0);

		value.resize(size);
		memcpy(&value[0], &result[0], size * sizeof(uint64_t));
		return *this;
	}

	BigInt& MultiplyBase(size_t times) {
		value.insert(value.begin(), times, 0);
		return *this;
	}

	BigInt& KMul(const BigInt& other) {
		const size_t m = (value.size() < other.value.size() ? value.size() : other.value.size());
		const size_t m2 = m >> 1;

		BigInt high1, low1, high2, low2;
		BigInt z0, z1, z2;

		SplitAt(*this, m2, high1, low1);
		SplitAt(other, m2, high2, low2);

		static bool workerId0, workerId1, workerId2;

		if (!MULTI_THREAD || m < MULTI_THREAD_LIMIT || (workerId0 || workerId1 || workerId2)) {
			z0 = low1 * low2;
			z1 = (low1 + high1) * (low2 + high2);
			z2 = high1 * high2;
		}
		else {
			workerId0 = true;
			workerId1 = true;
			workerId2 = true;

			auto lambda0 = [&] { z0 = low1 * low2; workerId0 = false; };
			auto lambda1 = [&] { z1 = (low1 + high1) * (low2 + high2); workerId1 = false; };
			auto lambda2 = [&] { z2 = high1 * high2; workerId2 = false; };

			std::thread worker0(lambda0);
			std::thread worker1(lambda1);
			std::thread worker2(lambda2);

			worker1.join();
			worker0.join();
			worker2.join();
		}

		z1.Subtract(z2);
		z1.Subtract(z0);
		z2.MultiplyBase(m2 << 1);
		z1.MultiplyBase(m2);
		z2.Add(z1);
		z2.Add(z0);

		value = std::move(z2.value);

		return *this;
	}

	void SplitAt(const BigInt& num, const size_t n, BigInt& high, BigInt& low) {
		std::vector<var_type> lowValue(num.value.begin(), num.value.begin() + n);
		std::vector<var_type> highValue(num.value.end() - (num.value.size() - n), num.value.end());

		low.value = std::move(lowValue);
		high.value = std::move(highValue);
	}

	BigInt& Pow(const size_t p) {
		if (p == 0) {
			*this = 1;
			return *this;
		}

		BigInt result(*this);
		for (size_t i = 0; i < p - 1; ++i) {
			result *= *this;
		}
		*this = result;
		return *this;
	}

	BigInt& Sqrt() {
		BigInt s = 0, t, u = *this;
		do {
			s = u;
			t = s + (*this) / s;
			u = t >> 1;
		} while (u < s);

		value = std::move(s.value);
		return *this;
	}

	BigInt& ModPow(const BigInt& e, const BigInt& m) {
		size_t bitSize = e.BitSize();
		BigInt result = *this % m;

		for (size_t i = 1; i < bitSize; ++i) {
			result *= result;
			result %= m;
			if (e.TestBit(bitSize - (i + 1))) {
				result *= *this;
				result %= m;
			}
		}

		value = std::move(result.value);
		return *this;
	}

	BigInt& MontDivR(size_t rsize) {
		if (value.size() > rsize) {
			value.erase(value.begin(), value.begin() + rsize);
		}
		else {
			*this = 0;
		}
		return *this;
	}

	BigInt& MontModR(size_t rsize) {
		if (value.size() > rsize) {
			value.erase(value.end() - (value.size() - rsize), value.end());
		}
		return *this;
	}

	BigInt& MontReduce(size_t rsize, const BigInt& m, const BigInt& mprime) {
		BigInt n = *this;
		n.MontModR(rsize);

		n *= mprime;
		n.MontModR(rsize);

		n *= m;

		Add(n);
		MontDivR(rsize);

		if (*this >= m) {
			*this -= m;
		}

		return *this;
	}

	BigInt& MontPow(const BigInt& e, const BigInt& m) {
		size_t rsize = m.value.size();
		BigInt r, rinv = 1, mprime = 0;

		mprime.value.resize(m.value.size());
		r.value.resize(rsize > 1 ? rsize : 2);
		r.value[1] = 1;

		for (size_t i = 0; i < rsize - 1; ++i) {
			r <<= EXPONENT;
		}

		for (size_t i = 0; i < rsize * EXPONENT; ++i) {
			if ((rinv[0] & 1) == 0) {
				rinv >>= 1;
				mprime >>= 1;
			}
			else {
				rinv.Add(m);
				rinv >>= 1;
				if (i != 0)
					mprime >>= 1;
				mprime.SetBit(rsize * EXPONENT - 1);
			}

		}

		MontPow(e, m, mprime, r, rsize);
	}

	BigInt& MontPow(const BigInt& e, const BigInt& m, const BigInt& mprime, const BigInt r, const size_t rsize) {
		const uint8_t k = (e.BitSize() < 512 ? 4 : 5);
		std::vector<BigInt> g((uint64_t)1 << k);

		g[0] = *this * ((r * r) % m);
		g[0].MontReduce(rsize, m, mprime);
		BigInt g2 = g[0];

		g2 = g2 * g2;
		g2.MontReduce(rsize, m, mprime);

		for (size_t i = 1; i < g.size(); ++i) {
			g[i] = g[i - 1];
			g[i] *= g2;
			g[i].MontReduce(rsize, m, mprime);
		}

		size_t bitSize = e.BitSize();
		BigInt result = r % m;
		int64_t i = bitSize - 1;
		while (i >= 0) {
			if (e.TestBit(i)) {
				uint64_t l = (i - k + 1 > 0 ? i - k + 1 : 0);
				while (!e.TestBit(l)) {
					l++;
				}
				for (size_t j = 0; j < i - l + 1; ++j) {
					result *= result;
					result.MontReduce(rsize, m, mprime);
				}

				uint64_t ndx = 0;
				for (int64_t j = i; j >= l; --j) {
					if (j < 0) {
						break;
					}

					ndx <<= 1;
					if (e.TestBit(j)) {
						ndx++;
					}
				}

				ndx >>= 1;
				result *= g[ndx];
				result.MontReduce(rsize, m, mprime);
				i = l - 1;
			}
			else {
				result *= result;
				result.MontReduce(rsize, m, mprime);
				i--;
			}
		}

		result.MontReduce(rsize, m, mprime);

		value = std::move(result.value);
		return *this;
	}

	BigInt& operator/=(const BigInt& other) {
		sign *= other.sign;
		Divide(other);
		return *this;
	}

	BigInt& Divide(const BigInt& other, BigInt* r = nullptr) {
		if (value.size() == 1 && value[0] == 0) {
			if (r != NULL) {
				r->sign = 1;
				r->value.resize(1);
				r->value[0] = 0;
			}
			return *this;
		}

		if (other.value.size() > 1)
			return DivisionD(other, r);

		return ShortDiv(other, r);
	}

	BigInt& ShortDiv(const BigInt& other, BigInt* r = nullptr) {
		if (*this < other) {
			if (r != nullptr) {
				r->value = std::move(value);
			}
			else {
				*this = 0;
			}

			return *this;
		}

		size_t n = value.size();
		uint64_t d = other.value[0];

		std::vector<var_type> q(n);
		uint64_t t, k = 0;

		for (int64_t i = n - 1; i >= 0; --i) {
			if (value[i] + k >= d) {
				t = value[i] + k;
				q[i] = t / d;
				k = (t % d) << EXPONENT;
			}
			else {
				k = (value[i] % d) << EXPONENT;
			}
		}
		if (r != nullptr) {
			*r = k >> EXPONENT;
		}

		value = std::move(q);

		Clean();
		return *this;
	}

	BigInt& DivisionD(const BigInt& other, BigInt* r = nullptr) {
		if (*this < other) {
			if (r != nullptr) {
				r->value = std::move(value);
			}
			else {
				*this = 0;
			}

			return *this;
		}

		uint64_t n = other.value.size();
		uint64_t m = value.size() - n;
		uint64_t qhat, rhat;
		BigInt un, vn;

		uint8_t shift = LeadingZeros(other.value[n - 1]);

		vn = other;
		un = *this;
		un.value.push_back(0);
		vn <<= shift;
		un <<= shift;

		std::vector<var_type> q;
		q.resize(m + 1);

		for (int64_t j = m; j >= 0; --j) {
			qhat = (un[j + n] * BASE + un[j + n - 1]) / vn[n - 1];
			rhat = (un[j + n] * BASE + un[j + n - 1]) % vn[n - 1];
			do {
				if (qhat >= BASE || qhat * vn[n - 2] > BASE * rhat + un[j + n - 2]) {
					qhat--;
					rhat += vn[n - 1];
				}
				else {
					break;
				}
			} while (rhat < BASE);

			int64_t borrow = 0;
			int64_t t = 0;
			for (size_t i = 0; i < n; ++i) {
				uint64_t p = qhat * vn[i];
				t = un[i + j] - borrow - (p & (BASE - 1));			
				un[i + j] = t & (BASE - 1);
				borrow = (p >> EXPONENT) - (t >> EXPONENT);
			}
			t = un[j + n] - borrow;
			un[j + n] = t;

			q[j] = qhat;             
			if (t < 0) {           
				q[j]--;       
				borrow = 0;
				for (size_t i = 0; i < n; ++i) {
					t = un[i + j] + vn[i] + borrow;
					un[i + j] = t & (BASE - 1);
					borrow = t >> EXPONENT;
				}
				un[j + n] += borrow;
			}
		}

		value = std::move(q);
		Clean();

		if (r != nullptr) {
			un >>= shift;
			r->value = std::move(un.value);
			r->Clean();
		}

		return *this;
	}

	bool IsPrime() {
		const BigInt one(1);
		const BigInt two(2);
		const BigInt three(3);
		
		if (*this <= one || *this % two == 0 || *this % three == 0)
			return false;

		for (size_t i = 2; i * i <= *this; i += 6) {
			if (*this % i == 0 || *this % (i + 2) == 0)
				return false;
		}

		return true;
	}

	bool IsEven() const {
		return value[0] % 2 == 0;
	}

	bool IsZero() const {
		return value.size() <= 1 && value[0] == 0;
	}

	bool IsNegative() const {
		return (sign == 1 ? -1 : 1);
	}

	size_t Size() const {
		return value.size();
	}

	static uint8_t LeadingZeros(var_type x) {
		uint8_t n = 0;
		while (x <= (BASE - 1) / 2) {
			x <<= 1;
			n++;
		}

		return n;
	}

	size_t BitSize() const {
		uint64_t count = 0;
		var_type high = value[value.size() - 1];
		while (high != 0) {
			high >>= 1;
			count += 1;
		}

		return (value.size() - 1) * EXPONENT + count;
	}

	bool TestBit(size_t k) const {
		size_t num = k / EXPONENT;
		size_t i = k - (num * EXPONENT);
		return value[num] & ((uint64_t)1 << i);
	}

	void SetBit(size_t k) {
		size_t num = k / EXPONENT;
		size_t i = k - (num * EXPONENT);
		value[num] |= value[num] | ((uint64_t)1 << i);
	}

	uint64_t ToInt() const {
		int64_t result = 0;
		int64_t power = 1;
		int64_t base = BASE;
		for (size_t i = 0; i < value.size(); ++i) {
			result += value[i] * power;
			power *= base;
		}

		return result * sign;
	}

	BigInt& operator<<=(const uint32_t shift) {
		if (shift <= 0)
			return *this;

		uint64_t k, t;
		k = 0;
		for (size_t i = 0; i < value.size(); ++i) {
			t = (uint64_t)value[i] >> (EXPONENT - shift);
			value[i] = (((uint64_t)value[i] << shift) | k) & (BASE - 1);
			k = t;
			if (i == value.size() - 1 && k != 0)
				value.push_back(0);
		}

		return *this;
	}

	friend BigInt operator<<(const BigInt& lhs, const uint8_t shift) {
		BigInt result(lhs);
		result <<= shift;
		return result;
	}

	BigInt& operator>>=(const uint32_t shift) {
		if (shift <= 0)
			return *this;

		uint64_t k, t;

		k = 0;
		for (int64_t i = value.size() - 1; i >= 0; --i) {
			t = (uint64_t)value[i] << (EXPONENT - shift);
			value[i] = ((value[i] >> shift) | k) & (BASE - 1);
			k = t;
		}
		Clean();
		return *this;
	}

	friend BigInt operator>>(const BigInt& lhs, const uint8_t shift) {
		BigInt result(lhs);
		result >>= shift;
		return result;
	}

	var_type& operator[](const size_t index) {
		return value[index];
	}

	friend BigInt operator/(const BigInt& lhs, const BigInt& rhs) {
		BigInt result(lhs);
		result /= rhs;
		return result;
	}

	BigInt& operator%=(const BigInt& other) {
		BigInt r;
		Divide(other, &r);
		value = std::move(r.value);
		return *this;
	}

	friend BigInt operator%(const BigInt& lhs, const BigInt& rhs) {
		BigInt result(lhs);
		result %= rhs;
		return result;
	}

	BigInt& operator=(const BigInt& other) {
		value = other.value;
		sign = other.sign;
		return *this;
	}

	friend bool operator==(const BigInt& lhs, const BigInt& rhs) {
		const size_t a = lhs.value.size();
		const size_t b = rhs.value.size();

		if (a != b)
			return false;
		for (size_t i = 0; i < a; ++i) {
			if (lhs.value[i] != rhs.value[i])
				return false;
		}

		return true;
	}

	friend bool operator!=(const BigInt& lhs, const BigInt& rhs) {
		return !(lhs == rhs);
	}

	friend bool operator>(const BigInt& lhs, const BigInt& rhs) {
		size_t a = lhs.value.size();
		size_t b = rhs.value.size();

		if (a > b)
			return true;
		if (b > a)
			return false;
		for (size_t i = 0; i < a; ++i) {
			var_type v1 = lhs.value[a - (i + 1)];
			var_type v2 = rhs.value[a - (i + 1)];
			if (v1 > v2)
				return true;
			else if (v1 < v2)
				return false;
		}

		return false;
	}

	friend bool operator>=(const BigInt& lhs, const BigInt& rhs) {
		if (lhs > rhs || lhs == rhs)
			return true;

		return false;
	}

	friend bool operator<(const BigInt& lhs, const BigInt& rhs) {
		if (lhs > rhs)
			return false;
		if (lhs == rhs)
			return false;

		return true;
	}

	friend bool operator<=(const BigInt& lhs, const BigInt& rhs) {
		if (lhs < rhs || lhs == rhs)
			return true;

		return false;
	}

	static BigInt GCD(BigInt a, BigInt b) {
		BigInt temp, t = 1;

		while (a.IsEven() && b.IsEven()) {
			t <<= 1;
			a >>= 1;
			b >>= 1;
		}
		while (a.IsEven()) {
			a >>= 1;
		}
		while (b.IsEven()) {
			b >>= 1;
		}
		while (a != b) {
			temp = (a < b ? a : b);
			a.Difference(b);
			b = temp;
			while (a.IsEven()) {
				a >>= 1;
			}
		}

		return t * a;
	}

	static void EGCD(const BigInt& a, const BigInt& b, BigInt* gcd, BigInt* co1, BigInt* co2) {
		BigInt old_r, r;
		BigInt old_s, s;
		BigInt old_t, t;
		BigInt q, temp;

		old_r = a;
		r = b;
		old_s = 1;
		s = 0;
		old_t = 0;
		t = 1;

		while (r != 0) {
			q = old_r / r;

			temp = r;
			BigInt testk = q * temp;
			r = old_r - testk;
			old_r = temp;

			temp = s;
			s = old_s - q * temp;
			old_s = temp;

			temp = t;
			t = old_t - q * temp;
			old_t = temp;
		} 

		if (co1 != NULL)
			(*co1) = old_s;
		if (co2 != NULL)
			(*co2) = old_t;
		if(gcd != NULL)
			(*gcd) = old_r;
	}

	static BigInt LCM(const BigInt& a, const BigInt& b) {
		BigInt result = a * b;
		result.sign = 1;
		result /= GCD(a, b);
		return result;
	}

	static BigInt ModInv(const BigInt& a, const BigInt& b) {
		BigInt gcd, x; 

		EGCD(a, b, &gcd, &x, NULL);
		return (x % b + b) % b;
	}

	static BigInt PollardRho(const BigInt& n) {
		if (n == 1) return n;
		if (n % 2 == 0) return 2;

		BigInt x = 2, y = 2, d = 1, diff;
		const BigInt one = 1;
		uint64_t i = 0;

		while (d == 1) {
			x = x * x;
			x.Add(one);
			x %= n;

			y = y * y;
			y.Add(one);
			y %= n;
			y = y * y;
			y.Add(one);
			y %= n;

			diff = (x > y ? x : y);
			diff = diff.Subtract((x > y ? y : x));
			d = GCD(diff, n);
		}
		if (d == n)
			return -1;
		else
			return d;
	}

	static BigInt RndNBit(size_t n) {
		std::uniform_int_distribution<uint16_t> dist01(0, 1);
		BigInt rnd;
		rnd.value.resize(n / EXPONENT + 1);
		rnd.SetBit(n - 1);
		for (size_t i = 0; i < n; ++i) {
			if (dist01(RndEngine)) {
				rnd.SetBit(i);
			}
		}

		return rnd;
	}

	static bool MillerRabin(const BigInt& n, int k) {
		std::uniform_int_distribution<uint64_t> rnd(2, ((n - 2) >= INT64_MAX ? INT64_MAX : (n.ToInt() - 2)));

		uint32_t s = 0;
		BigInt d = n - 1;
		BigInt a, x;

		bool tryAgain = false;

		while (d.IsEven()) {
			s++;
			d >>= 1;
		}
		for (size_t i = 0; i < k; ++i) {
			a = rnd(RndEngine);
			x = a.ModPow(d, n);
			if (x == 1 || x == n - 1) {
				continue;
			}
			for (size_t r = 0; r < s; ++r) {
				x.ModPow(2, n);
				if (x == 1) {
					return false;
				}
				else if (x == n - 1) {
					tryAgain = true;
					break;
				}
			}
			if (!tryAgain)
				return false;
			tryAgain = false;
		}
		return true;
	}

	static bool MillerRabinMont(const BigInt& n, int k) {
		std::uniform_int_distribution<uint64_t> rnd(2, ((n - 2) >= INT64_MAX ? INT64_MAX : (n.ToInt() - 2)));

		uint32_t s = 0;
		BigInt d = n - 1;
		BigInt a, x;

		bool tryAgain = false;

		while (d.IsEven()) {
			s++;
			d >>= 1;
		}

		size_t rsize = n.value.size();
		BigInt r, rinv = 1, mprime = 0;

		mprime.value.resize(n.value.size());
		r.value.resize(rsize > 1 ? rsize : 2);
		r.value[1] = 1;

		for (size_t i = 0; i < rsize - 1; ++i) {
			r <<= EXPONENT;
		}

		for (size_t i = 0; i < rsize * EXPONENT; ++i) {
			if ((rinv[0] & 1) == 0) {
				rinv >>= 1;
				mprime >>= 1;
			}
			else {
				rinv.Add(n);
				rinv >>= 1;
				if (i != 0)
					mprime >>= 1;
				mprime.SetBit(rsize * EXPONENT - 1);
			}

		}

		for (size_t i = 0; i < k; ++i) {
			a = rnd(RndEngine);
			x = a.MontPow(d, n, mprime, r, rsize);

			if (x == 1 || x == n - 1) {
				continue;
			}

			for (size_t r = 0; r < s; ++r) {
				x.ModPow(2, n);
				if (x == 1) {
					return false;
				}
				else if (x == n - 1) {
					tryAgain = true;
					break;
				}
			}
			if (!tryAgain)
				return false;
			tryAgain = false;
		}
		return true;
	}

	std::string ToString() const {
		if (*this == 0) {
			return "0";
		}

		uint64_t m = 0;
		BigInt a = *this;

		const BigInt ten = 10;

		std::string s;
		while (a != 0) {
			s += (a % ten).value[0] + '0';
			
			a /= ten;
			m++;
		}
		if (sign == -1)
			s += '-';

		std::reverse(s.begin(), s.end());
		return s;
	}

	void Print(bool base10) const {
		if (base10) {
			std::cout << ToString() << "\n\n";
		}
		else {
			if (sign == -1)
				std::cout << "-";

			for (const auto& i : value) {
				std::cout << i;
			}
			std::cout << "\n\n" << value.size() << " digits" << std::endl;
		}
	}
};