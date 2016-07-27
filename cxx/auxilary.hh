namespace std{
namespace
{

  // Code from boost
  // Reciprocal of the golden ratio helps spread entropy
  //     and handles duplicates.
  // See Mike Seymour in magic-numbers-in-boosthash-combine:
  //     http://stackoverflow.com/questions/4948780

  template <class T>
  inline void hash_combine(std::size_t& seed, T const& v)
  {
    seed ^= hash<T>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
  }

  // Recursive template code derived from Matthieu M.
  template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
    struct HashValueImpl
    {
      static void apply(size_t& seed, Tuple const& tuple)
      {
	HashValueImpl<Tuple, Index-1>::apply(seed, tuple);
	hash_combine(seed, get<Index>(tuple));
      }
    };

  template <class Tuple>
  struct HashValueImpl<Tuple,0>
  {
    static void apply(size_t& seed, Tuple const& tuple)
    {
      hash_combine(seed, get<0>(tuple));
    }
  };
}

  template <typename ... TT>
  struct hash<std::tuple<TT...>> 
  {
    size_t
      operator()(std::tuple<TT...> const& tt) const
    {                                              
      size_t seed = 0;                             
      HashValueImpl<std::tuple<TT...> >::apply(seed, tt);    
      return seed;                                 
    }                                              

  };
}

static int product_prom[204] = { 897,1124,1289,2320,3303,3305,3621,3678,4102,4259,4773,5901,30172,30173,30370,30372,30403,30505,30506,30507,30531,30532,30533,30543,30544,30569,30572,30575,30706,30707,30888,30904,30906,30916,30954,30958,31080,31082,31083,31095,31098,31100,31103,31118,31119,31120,31142,31145,31283,31284,31320,31322,31457,31778,32053,32092,32134,32220,32221,32222,32224,32322,32410,32863,32864,32939,32940,32941,32999,33021,33054,33074,33862,33863,34044,34048,34429,34430,34486,34487,34775,34776,34943,34944,34945,34992,34993,35054,35074,35176,35177,35179,35180,35182,35183,35185,35186,35212,35213,35215,35217,35293,35294,35301,35303,35304,35305,35306,35307,35308,35309,35310,35438,35473,35557,35558,35561,35578,35595,35660,35744,35761,35763,36270,36322,36323,36529,36544,36545,36547,36550,36551,36553,36554,36744,36773,36806,36873,36927,36928,36929,36930,36931,36989,37018,37360,37361,37362,37363,37379,37402,37403,37404,37516,37519,37520,37523,37569,37570,37573,37574,37575,37576,37577,37578,37579,37580,37581,40217,40929,40930,40931,40942,40944,40945,41830,41935,42034,42110,44576,44950,44951,45111,45112,45141,45407,46170,46171,47661,47662,47676,47886,47887,47950,48106,48582,48597,48598,48874,48876,49492,49781,49835,49920 };
