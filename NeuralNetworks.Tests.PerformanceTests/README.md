# Latest Performance Benchmarks 

``` ini

BenchmarkDotNet=v0.10.9, OS=Mac OS X 10.12
Processor=Intel Core i5-4278U CPU 2.60GHz (Haswell), ProcessorCount=4
.NET Core SDK=2.0.0
  [Host]     : .NET Core 2.0.0 (Framework 4.6.00001.0), 64bit RyuJIT
  DefaultJob : .NET Core 2.0.0 (Framework 4.6.00001.0), 64bit RyuJIT


```
 |                                                  Method |    Mean |    Error |   StdDev |  Median |
 |-------------------------------------------------------- |--------:|---------:|---------:|--------:|
 |  BackPropagationTrainingMultiThreadedFiveThousandEpochs | 2.982 s | 0.0590 s | 0.1522 s | 2.913 s |
 | BackPropagationTrainingSingleThreadedFiveThousandEpochs | 3.037 s | 0.0028 s | 0.0026 s | 3.037 s |


 ``` ini

BenchmarkDotNet=v0.10.9, OS=Mac OS X 10.12
Processor=Intel Core i5-4278U CPU 2.60GHz (Haswell), ProcessorCount=4
.NET Core SDK=2.0.0
  [Host]     : .NET Core 2.0.0 (Framework 4.6.00001.0), 64bit RyuJIT
  Job-JYVQDG : .NET Core 2.0.0 (Framework 4.6.00001.0), 64bit RyuJIT

LaunchCount=10  RunStrategy=Monitoring  TargetCount=20  
WarmupCount=5  

```
 |                              Method |     Mean |     Error |    StdDev |   Median |
 |------------------------------------ |---------:|----------:|----------:|---------:|
 | FeedForwardPredictionSingleThreaded | 1.697 ms | 0.1621 ms | 0.6863 ms | 1.393 ms |
 |  FeedForwardPredictionMultiThreaded | 1.335 ms | 0.1795 ms | 0.7600 ms | 1.146 ms |