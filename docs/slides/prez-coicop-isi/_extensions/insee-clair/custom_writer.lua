-- Fonction pour centrer les images
function Image(img)
    -- Retourne l'image avec un style de paragraphe centré
    return pandoc.Para({
        pandoc.Image(img.attr, img.caption)
    }) ..
    pandoc.RawBlock('openxml', '<w:pPr><w:jc w:val="center"/></w:pPr>')
end

-- Fonction pour personnaliser les puces des listes
function BulletList(list)
    -- Retourne la liste avec des styles de puces personnalisés
    return pandoc.BulletList(pandoc.utils.map(
        function(item)
            if #item > 0 and item[1].t == "Plain" then
                -- Appliquer un style de paragraphe aux éléments de la liste
                return pandoc.Para(pandoc.Strong({pandoc.Str("• ")})) .. item
            else
                return item
            end
        end,
        list.content
    ))
end

-- Fonction pour personnaliser les listes numérotées
function OrderedList(list)
    -- Retourne la liste numérotée avec des styles personnalisés
    return pandoc.OrderedList(list.attr)(
        pandoc.utils.map(
            function(item)
                if #item > 0 and item[1].t == "Plain" then
                    -- Appliquer un style de paragraphe aux éléments de la liste
                    return pandoc.Para(pandoc.Strong({pandoc.Str("1. ")})) .. item
                else
                    return item
                end
            end,
            list.content
        )
    )
end

-- Fonction pour appliquer un style de paragraphe centré aux images
function Para(el)
    local has_image = false
    for _, elem in ipairs(el.content) do
        if elem.t == "Image" then
            has_image = true
            break
        end
    end
    if has_image then
        -- Ajouter un style de paragraphe centré
        el.attributes = el.attributes or {}
        el.attributes["custom-style"] = "ImageCentrée"
    end
    return el
end

-- Fonction pour appliquer des styles de puces aux listes
function Doc(doc)
    -- Ajouter des métadonnées pour les styles de liste
    doc.meta["list-styles"] = pandoc.MetaMap({
        ["level-1"] = pandoc.MetaInlines({pandoc.Str("-")}),
        ["level-2"] = pandoc.MetaInlines({pandoc.Str("•")}),
        ["level-3"] = pandoc.MetaInlines({pandoc.Str("-")})
        ["level-4"] = pandoc.MetaInlines({pandoc.Str("•")})
        ["level-5"] = pandoc.MetaInlines({pandoc.Str("•")})
        ["level-6"] = pandoc.MetaInlines({pandoc.Str("•")})
    })
    return doc
end
