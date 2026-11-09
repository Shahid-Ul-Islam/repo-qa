using System;

namespace Demo.Services
{
    /// <summary>Repository of orders.</summary>
    public class OrderRepository
    {
        private readonly int _limit = 5;

        public OrderRepository(int limit)
        {
            _limit = limit;
        }

        /// <summary>Finds an order.</summary>
        public string FindOrder(int id)
        {
            return $"order-{id}";
        }
    }

    public interface IClock
    {
        DateTime Now();
    }
}
