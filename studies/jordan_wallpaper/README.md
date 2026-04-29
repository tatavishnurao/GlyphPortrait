# Jordan Wallpaper Study

This folder contains intentionally overfit, target-aware reconstruction code for
reverse-engineering a Jordan-style typographic wallpaper.

It is separate from the reusable `glyphforge/` engine on purpose:

- `glyphforge/` = reusable core for generic portrait typography rendering
- `studies/jordan_wallpaper/` = research workflow for one specific visual target

## Scope

- target decomposition (subject/jersey/face)
- right-side 16:9 composition
- multi-pass typography reconstruction
- anchor word layer
- diagnostic metrics

This study code is not meant to be a general portrait generator yet.
