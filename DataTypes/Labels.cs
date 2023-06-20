using Newtonsoft.Json;

/// <summary>
/// The <see cref="Labels"/> structs specifies the number of labels in which images have been classified.
/// </summary>
[System.Serializable]
public readonly struct Labels
{
    /// <value>The number of artists images have been classified by.</value>
    [JsonProperty] public int Artists { get; }

    /// <value>The number of specific characters images have been classified by.</value>
    [JsonProperty] public int Characters { get; }

    /// <value>The number of races images have been classified by.</value>
    /// Note: This network was originally designed using images of fantasy characters. In this case race refers to fantasy race.
    [JsonProperty] public int Races { get; }

    /// <value>The number of image sizes images have been classified by.</value>
    [JsonProperty] public int Sizes { get; }

    /// <value>The number of art styles images have been classified by.</value>
    [JsonProperty] public int Styles { get; }

    /// <value>The number of miscellaneous tags images have been classified by.</value>
    [JsonProperty] public int Tags { get; }
}