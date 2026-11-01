/*+ HashJoin(k mk mc cn t)
 MergeJoin(k mk mc cn)
 HashJoin(k mk mc)
 HashJoin(k mk)
 IndexScan(k)
 IndexScan(mk)
 IndexScan(mc)
 IndexScan(cn)
 SeqScan(t)
 Leading(((((k mk) mc) cn) t)) */
SELECT MIN(t.title) AS movie_title
FROM company_name AS cn,
     keyword AS k,
     movie_companies AS mc,
     movie_keyword AS mk,
     title AS t
WHERE cn.country_code ='[de]'
  AND k.keyword ='character-name-in-title'
  AND cn.id = mc.company_id
  AND mc.movie_id = t.id
  AND t.id = mk.movie_id
  AND mk.keyword_id = k.id
  AND mc.movie_id = mk.movie_id;

