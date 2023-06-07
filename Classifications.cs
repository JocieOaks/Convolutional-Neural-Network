using System.Collections.Generic;
using Newtonsoft.Json;

[System.Serializable]
public struct Classifications
{
    public Classifications(int names, int races, int tags, int artists, int styles, int sizes)
    {
        Names = names;
        Races = races;
        Tags = tags;
        Artists = artists;
        Styles = styles;
        Sizes = sizes;
    }

    [JsonProperty] public int Artists { get; }
    [JsonProperty] public int Names { get; }
    [JsonProperty] public int Races { get; }
    [JsonProperty] public int Sizes { get; }
    [JsonProperty] public int Styles { get; }
    [JsonProperty] public int Tags { get; }
}
