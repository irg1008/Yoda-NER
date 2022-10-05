SELECT TOP (100000)
    [Title] as title
      , [Brand] as brand
      , [Color] as color
      , [Size] as size
      -- , [Material] as material
FROM [lighthousefeed_prod].[dbo].[Products]
ORDER BY RAND ()  