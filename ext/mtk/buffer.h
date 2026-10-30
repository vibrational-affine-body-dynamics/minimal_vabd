#pragma once
#include <vector>
#include "common.cuh"
// #include <vector>
namespace Shay
{
	template <class T>
	class VtBuffer
	{
	public:
		VtBuffer() {}

		VtBuffer(uint size)
		{
			resize(size);
		}

		// VtBuffer(const VtBuffer&) = delete;
		// VtBuffer(const std::vector<T> &host_vector)
		// {
		// 	push_back(host_vector);
		// }

		VtBuffer &operator=(const VtBuffer &) = delete;

		operator T *() const { return m_buffer; }

		~VtBuffer()
		{
			destroy();
		}

		void zero_()
		{
			cudaMemset(m_buffer, 0, size() * sizeof(T));
		}

		void fill_(int val)
		{
			cudaMemset(m_buffer, 1, size() * sizeof(T));
		}
		__host__ __device__ T &operator[](size_t index)
		{
			// assert(m_buffer);
			// assert(index < m_count);
			return m_buffer[index];
		}

		__host__ __device__ const T &operator[](size_t index) const
		{
			// assert(m_buffer);
			// assert(index < m_count);
			return m_buffer[index];
		}

		__host__ __device__ size_t size() const { return m_count; }

		void push_back(const T &t)
		{
			reserve(m_count + 1);
			m_buffer[m_count++] = t;
		}

		void push_back(size_t newCount, const T &val)
		{
			for (int i = 0; i < newCount; i++)
			{
				push_back(val);
			}
		}

		void copy(const std::vector<T> &data)
		{
			resize(data.size());
			cudaMemcpy(m_buffer, data.data(), sizeof(T) * data.size(), cudaMemcpyDefault);
		}

		std::vector<T> get() const
		{
			std::vector<T> ret;
			ret.resize(size());
			cudaMemcpy(ret.data(), m_buffer, sizeof(T) * size(), cudaMemcpyDefault);
			return ret;
		}
		// void push_back(const std::vector<T> &data)
		// {
		// 	size_t offset = m_count;
		// 	resize(m_count + data.size());
		// 	memcpy(m_buffer + offset, data.data(), data.size() * sizeof(T));
		// }

		void reserve(size_t minCapacity)
		{
			if (minCapacity > m_capacity)
			{
				// growth factor of 1.5
				const size_t newCapacity = minCapacity * 3 / 2;
#ifdef __CUDARUN__

				T *newBuf = VtAllocBuffer<T>(newCapacity);

#else
				T *newBuf = new T[newCapacity];
#endif
				// copy contents to new buffer
				if (m_buffer)
				{
					cudaMemcpy(newBuf, m_buffer, m_count * sizeof(T), cudaMemcpyDefault);
#ifdef __CUDARUN__
					VtFreeBuffer(m_buffer);
#else
					delete[] m_buffer;
#endif
				}

				// swap
				m_buffer = newBuf;
				m_capacity = newCapacity;
			}
		}

		void resize(size_t newCount)
		{
			reserve(newCount);
			m_count = newCount;
		}

		void resize(size_t newCount, const T &val)
		{
			const size_t startInit = m_count;
			const size_t endInit = newCount;

			resize(newCount);

			// init any new entries
			for (size_t i = startInit; i < endInit; ++i)
				m_buffer[i] = val;
		}

		T *data() const
		{
			return m_buffer;
		}

		void destroy()
		{
			if (m_buffer != nullptr)
			{

				// for (size_t i = 0; i < m_count; i++)
				//{
				//	m_buffer[i].~T();
				// }
#ifdef __CUDARUN__

				VtFreeBuffer(m_buffer);
#else
				delete[] m_buffer;
#endif
			}
			m_count = 0;
			m_capacity = 0;
			m_buffer = nullptr;
		}

	private:
		size_t m_count = 0;
		size_t m_capacity = 0;
		T *m_buffer = nullptr;
	};

}; // namespace cuda

#ifdef __CUDARUN__
template <typename T>
using wp_array = Shay::VtBuffer<T>;
#else
template <typename T>
using wp_array = std::vector<T>;
#endif