using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


/// <summary>
/// The <see cref="FeatureAtlas"/> class is a class for collecting and organizing multiple <see cref="FeatureMap"/>s.
/// </summary>
public class FeatureAtlas
{
    public int Dimensions { get; }
    public int BatchSize { get; }

    private readonly FeatureMap[,] _featureMaps;

    
}

